//! Bounded ordered handoff: pipe backpressure overlaps the next GPU readback.
use anyhow::{Context, Result};
use std::{
    io::{Read, Write},
    process::{Child, ChildStderr, ChildStdin},
    sync::mpsc,
    thread,
};

pub(crate) struct Writer {
    sender: Option<mpsc::Sender<(Vec<u8>, mpsc::Sender<Result<Vec<u8>, String>>)>>,
    worker: Option<thread::JoinHandle<std::io::Result<()>>>,
}
impl Writer {
    pub(crate) fn new(mut input: ChildStdin) -> Result<Self> {
        let (sender, receiver) =
            mpsc::channel::<(Vec<u8>, mpsc::Sender<Result<Vec<u8>, String>>)>();
        let worker = thread::Builder::new()
            .name("vork-encode-pipe".into())
            .spawn(move || {
                let mut error = None;
                for (frame, returned) in receiver {
                    if error.is_none() {
                        error = input.write_all(&frame).err();
                    }
                    let result = match &error {
                        Some(error) => Err(error.to_string()),
                        None => Ok(frame),
                    };
                    // Complete every queued entry even after a broken pipe.
                    let _ = returned.send(result);
                }
                error.map_or(Ok(()), Err)
            })?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
        })
    }
    pub(crate) fn send(
        &self,
        frame: Vec<u8>,
        returned: mpsc::Sender<Result<Vec<u8>, String>>,
    ) -> Result<()> {
        self.sender
            .as_ref()
            .context("closed encoder")?
            .send((frame, returned))
            .context("encoder pipe writer stopped")
    }
    pub(crate) fn finish(&mut self) -> Result<()> {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("encoder pipe writer panicked"))??;
        }
        Ok(())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// Drain stderr concurrently so ffmpeg cannot block behind its diagnostic pipe.
pub(crate) struct Diagnostics(thread::JoinHandle<std::io::Result<Vec<u8>>>);
impl Diagnostics {
    pub(crate) fn new(mut stderr: ChildStderr) -> Result<Self> {
        Ok(Self(
            thread::Builder::new()
                .name("vork-ffmpeg-stderr".into())
                .spawn(move || {
                    let mut tail = std::collections::VecDeque::new();
                    let mut chunk = [0; super::constants::FFMPEG_STDERR_BYTES];
                    loop {
                        let count = stderr.read(&mut chunk)?;
                        if count == 0 {
                            break;
                        }
                        tail.extend(&chunk[..count]);
                        let excess = tail
                            .len()
                            .saturating_sub(super::constants::FFMPEG_STDERR_BYTES);
                        tail.drain(..excess);
                    }
                    Ok(tail.into_iter().collect())
                })?,
        ))
    }
    fn finish(self) -> Result<String> {
        let bytes = self
            .0
            .join()
            .map_err(|_| anyhow::anyhow!("ffmpeg stderr reader panicked"))??;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
pub(crate) fn finish(
    child: &mut Child,
    input: &mut Option<Writer>,
    stderr: &mut Option<Diagnostics>,
) -> Result<()> {
    let write_result = input.take().map(|mut writer| writer.finish()).transpose();
    let status = child.wait().context("wait for ffmpeg")?;
    let diagnostics = stderr
        .take()
        .map(Diagnostics::finish)
        .transpose()?
        .unwrap_or_default();
    if !status.success() {
        let message = format!(
            "ffmpeg failed ({status}): {}\n{}",
            super::constants::FFMPEG_REQUIREMENT,
            diagnostics.trim()
        );
        return match write_result {
            Err(error) => Err(error).context(message),
            Ok(_) => Err(anyhow::anyhow!(message)),
        };
    }
    write_result.context("write frames to ffmpeg")?;
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn broken_pipe_completes_every_queued_ring_entry() -> Result<()> {
        for script in ["exit 0", "exec 3<&0; sleep 0.1 <&3 & exit 0"] {
            check_broken_pipe(script)?;
        }
        Ok(())
    }

    fn check_broken_pipe(script: &str) -> Result<()> {
        let mut child = std::process::Command::new("sh")
            .args(["-c", script])
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        let input = child.stdin.take().unwrap();
        child.wait()?;
        let mut writer = Writer::new(input)?;
        // Waiting for this child does not exclude inherited readers: another
        // concurrent spawn can retain the read end between fork and exec.
        // Observe the broken pipe before asserting that its error is latched.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let (returned, completed) = mpsc::channel();
            writer.send(vec![0; 4096], returned)?;
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if completed.recv_timeout(remaining)?.is_err() {
                break;
            }
        }
        let completions: Vec<_> = (0..3)
            .map(|_| {
                let (returned, completed) = mpsc::channel();
                writer.send(vec![0; 4096], returned).unwrap();
                completed
            })
            .collect();
        for completed in completions {
            assert!(
                completed
                    .recv_timeout(std::time::Duration::from_secs(5))?
                    .is_err()
            );
        }
        assert!(writer.finish().is_err());
        Ok(())
    }
}
