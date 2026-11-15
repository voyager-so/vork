//! Recover enum identity erased by Slang's JSON scalar reflection. The pinned
//! compiler's AST supplies declarations and source spans, never a source regex.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputEnum {
    pub ty: String,
    pub cases: Vec<String>,
}

fn name(record: &str) -> Result<&str> {
    record
        .split("nameAndLoc : NameLoc{\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .context("Slang AST declaration name missing")
}
fn offset(record: &str) -> Result<usize> {
    record
        .split("loc : SourceLoc(")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .context("Slang AST source span missing")?
        .parse()
        .context("Slang AST source span changed")
}
fn parent(record: &str) -> Option<&str> {
    record
        .lines()
        .find_map(|line| line.trim().strip_prefix("parentDecl : "))
}

fn records(ast: &str) -> BTreeMap<&str, &str> {
    ast.split("\n}\n")
        .filter_map(|record| record.trim().split_once(" {\n"))
        .collect()
}

fn declarations(records: &BTreeMap<&str, &str>) -> Result<BTreeMap<String, Vec<String>>> {
    let mut enums = BTreeMap::new();
    for (id, record) in records {
        if !id.starts_with("EnumDecl:") { continue; }
        let mut cases: Vec<_> = records.iter()
            .filter(|(key, r)| key.starts_with("EnumCaseDecl:") && parent(r) == Some(*id))
            .map(|(_, r)| Ok((offset(r)?, name(r)?.to_owned())))
            .collect::<Result<_>>()?;
        cases.sort_by_key(|(position, _)| *position);
        if !cases.is_empty() {
            enums.insert(name(record)?.to_owned(), cases.into_iter().map(|(_, name)| name).collect());
        }
    }
    Ok(enums)
}

/// Missing enum members use the same supported-name diagnostic as authored enum
/// inputs. Slang supplies both the failed lookup and the actual declarations;
/// authored source is never scanned or rewritten.
pub(super) fn missing_members(directory: &Path, diagnostics: &str) -> Result<Option<String>> {
    let mut missing = BTreeSet::new();
    let mut errors = 0;
    let mut lookups = 0;
    let mut location = "";
    for line in diagnostics.lines() {
        if line.starts_with("error[") {
            if !line.starts_with("error[E30027]:") { return Ok(None); }
            errors += 1;
        }
        if let Some(path) = line.trim().strip_prefix("--> ") { location = path; }
        let Some((marker, message)) = line.trim().strip_prefix('|')
            .and_then(|s| s.trim().split_once(" '")) else { continue };
        if marker.is_empty() || !marker.chars().all(|c| c == '^' || c == '~') { continue; }
        let Some((member, ty)) = message.split_once("' is not a member of 'typeof(") else { continue };
        let Some(ty) = ty.strip_suffix(")'.") else { continue };
        missing.insert((location, ty, member));
        lookups += 1;
    }
    if errors == 0 || errors != lookups { return Ok(None); }
    let needed: BTreeSet<_> = missing.iter().map(|(_, ty, _)| *ty).collect();
    let mut enums = BTreeMap::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "slang-ast") { continue; }
        for (ty, cases) in declarations(&records(&read_compiler_text(path)?))? {
            if !needed.contains(ty.as_str()) { continue; }
            if enums.get(&ty).is_some_and(|previous| previous != &cases) { return Ok(None); }
            enums.insert(ty, cases);
        }
    }
    let mut messages = Vec::new();
    for (location, ty, member) in missing {
        let Some(cases) = enums.get(ty) else { return Ok(None) };
        messages.push(format!("{location}: {ty}: {}", vork_scene::diagnostics::unknown("member", member, cases.iter().map(String::as_str))));
    }
    Ok(Some(messages.join("; ")))
}

pub(super) fn reflect(directory: &Path, inputs: &mut InputLayout) -> Result<()> {
    let mut enums = BTreeMap::<String, Vec<String>>::new();
    let mut fields = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "slang-ast") {
            continue;
        }
        let ast = read_compiler_text(&path)?;
        let records = records(&ast);
        let Some((_, module)) = records
            .iter()
            .find(|(key, _)| key.starts_with("ModuleDecl:"))
        else {
            continue;
        };
        enums.extend(declarations(&records)?);
        let Some((input_id, _)) = records.iter().find(|(key, record)| {
            key.starts_with("StructDecl:") && name(record).ok() == Some("Inputs")
        }) else {
            continue;
        };
        let source_path = directory.join(format!("{}.slang", name(module)?));
        let source_path = if source_path.exists() {
            source_path
        } else {
            let location = module
                .lines()
                .find_map(|line| line.trim().strip_prefix("loc : SourceLoc("))
                .context("Slang AST module location missing")?;
            let file = location
                .split_once(") ")
                .and_then(|(_, s)| s.rsplit_once(':').map(|(file, _)| file))
                .context("Slang AST module source missing")?;
            PathBuf::from(file)
        };
        let source = fs::read_to_string(&source_path)
            .with_context(|| format!("read enum probe source {}", source_path.display()))?;
        let base = offset(module)?;
        for (id, record) in &records {
            if !id.starts_with("VarDecl:") || parent(record) != Some(*input_id) {
                continue;
            }
            let Some(expression) = record
                .split("type :")
                .nth(1)
                .and_then(|s| s.split_once('{').map(|(_, s)| s.trim()))
                .and_then(|s| s.lines().next())
                .map(|s| s.trim().trim_end_matches(','))
            else {
                continue;
            };
            let Some(expression) = records.get(expression) else {
                continue;
            };
            let start = offset(expression)?
                .checked_sub(base)
                .context("invalid Slang source span")?;
            let text = source
                .get(start..)
                .context("Slang enum source span outside module")?;
            let ty: String = text
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            fields.push((name(record)?.to_owned(), ty));
        }
    }
    for (name, ty) in fields {
        if let Some(cases) = enums.get(&ty) {
            if let Some(field) = inputs.fields.iter_mut().find(|f| f.name == name) {
                ensure!(
                    matches!(field.ty, Type::Int | Type::Uint),
                    "enum input {name} has unsupported storage"
                );
                let mut tokens = BTreeSet::new();
                ensure!(
                    cases
                        .iter()
                        .all(|case| tokens.insert(case.to_ascii_lowercase())),
                    "enum input {name}: case tokens collide after lowercasing"
                );
                field.enumeration = Some(InputEnum {
                    ty,
                    cases: cases.clone(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned Slang"]
    fn enum_tokens_preserve_explicit_values_and_author_defaults() -> Result<()> {
        let abi = Abi { aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        let source = "// enum values are intentionally nonsequential\nenum Finish { Matte = 17, Glossy = -3 }; struct Inputs { Finish finish=Finish::Glossy; }; Closure surface(ShadingContext ctx, Inputs inputs) { Closure c={}; c.base_weight=float(int(inputs.finish)); return c; }";
        let program = Compiler::discover().compile_surface(
            &abi,
            source,
            Path::new("enum-test.slang"),
            "enumtest",
        )?;
        let layout = &program.inputs;
        assert_eq!(
            layout.fields[0]
                .enumeration
                .as_ref()
                .context("enum metadata")?
                .cases,
            ["Matte", "Glossy"]
        );
        for (token, index) in [("matte", 0), ("glossy", 1)] {
            let words = layout.pack(
                &[("finish".into(), Value::from(token))].into(),
                |_, _| unreachable!(),
            )?;
            assert_eq!(words[layout.fields[0].offset], index);
        }
        for bad in [Value::from(17), Value::from("Glossy"), Value::from("other")] {
            assert!(
                layout
                    .pack(&[("finish".into(), bad)].into(), |_, _| unreachable!())
                    .is_err()
            );
        }
        assert_eq!(layout.pack(&BTreeMap::new(), |_, _| unreachable!())?[0], 0);
        Ok(())
    }
}
