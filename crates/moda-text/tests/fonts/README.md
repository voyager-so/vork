`HanTest.otf` is a test-only subset of NotoSansCJKjp-Regular.otf containing 漢 and 字.
Source: https://github.com/notofonts/noto-cjk/blob/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf
Copyright 2014-2021 Adobe (http://www.adobe.com/).
Licensed under SIL OFL 1.1; see OFL-CJK.txt.
Generated with fontTools subset, `--text=漢字`; name and copyright records are retained.

`NotoSans-WeightSubset.ttf` is a glyph subset of the repository's
`scenes/greenlight/text/assets/NotoSans.ttf` (Noto Sans, SIL OFL 1.1;
see `OFL-NotoSans.txt`). It retains the original variable axes and the characters
`HamburgefontsI` for the CSS weight, layout and ink-box regression tests.

Generated with fontTools 4.68.0:

```python
from fontTools import subset
from fontTools.ttLib import TTFont
font = TTFont("scenes/greenlight/text/assets/NotoSans.ttf")
options = subset.Options()
options.name_IDs = ["*"]
options.name_languages = ["*"]
options.name_legacy = True
subsetter = subset.Subsetter(options=options)
subsetter.populate(text="HamburgefontsI")
subsetter.subset(font)
font.save("crates/moda-text/tests/fonts/NotoSans-WeightSubset.ttf")
```

`Inter-ShapingSubset.ttf`, `NotoSans-ShapingSubset.ttf` and `Lato-ShapingSubset.ttf`
are test-only subsets for the HarfBuzz advance/cluster contract. Their sources are
Inter 24pt Regular (Inter Project Authors, https://github.com/rsms/inter), Noto Sans
(Noto Project Authors, https://github.com/notofonts/latin-greek-cyrillic), and Lato
Regular (Lukasz Dziedzic, https://www.latofonts.com/). All are SIL OFL 1.1;
see OFL-Inter.txt, OFL-NotoSans.txt and OFL-Lato.txt. The existing repository font
bytes were subset with fontTools 4.68.0 using all characters in `../harfbuzz-advances.json`.

Use the subset procedure above with `options.layout_features = ["*"]` and the five
reference strings joined as its text. All variable axes, copyright/license records,
and layout features remain; the GPOS extension lookups that trigger the regression
remain. Modified family/full/PostScript/unique names (name IDs 1, 3, 4, 6, 16, 17)
identify the fixtures as ShapingTestInter, ShapingTestNotoSans and ShapingTestLatin
(the latter avoids Lato's reserved font name). The goldens record the subset SHA-256
and independent HarfBuzz version; regeneration instructions are in the Rust test.

`Fraunces-SpacingSubset.ttf` is a test-only subset of Fraunces Regular,
Copyright 2020 The Fraunces Project Authors (https://github.com/undercasetype/Fraunces),
SIL OFL 1.1 (see `OFL-Fraunces.txt`). Source: the static Regular instance at
https://cdn.moda.app/msdf/fraunces/Fraunces[SOFT,WONK,opsz,wght]-Regular.ttf,
SHA-256 `dc127abf0182030060a444a754ec39e2475563fa3f4d13176b1957bdf0119586`.
The subset procedure above used fontTools 4.68.0, `text="AV To"` and
`options.layout_features = ["*"]`. Modified name IDs 1, 3, 4, 6, 16 and 17
identify it as SpacingTest. Advances and A/V kerning are preserved.
`../canvas2d_spacing.rs` records native Chrome 141 Canvas 2D measurements with
`fontKerning="auto"`, `letterSpacing="4px"`, `font="100px Fraunces"` and
font hinting disabled (`--font-render-hinting=none`): `AV` width 145.35,
space width 26.6, `AV AV` width 317.3, second word origin 171.95 (px).
These include trailing tracking, matching the production placement paths.
