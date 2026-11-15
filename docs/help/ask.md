ASK
vork ask SCENE QUESTIONS|-
Canvas documents support audio answers only.
QUESTIONS: a JSON list of addresses (- reads stdin); answers: a JSON list in the same order.
at: scene seconds, or a list of times for an answer at each.
Address keys: node with read transform|bounds|screen; field with a path; image with
read info; buffer with optional range or read count.
range is [first,count].
instance picks a copy, camera picks the view for screen.
screen gives projected bounds and inside, the fraction in frame; occlusion is not measured.
audio names a subtree; window: [start,end) is required in seconds; at offsets it (default 0).
Omit read for 48 kHz stereo samples; read loudness|spectrum for analysis.
file writes samples to WAV or analysis to JSON; requires one at time.
loudness: rms, peak, rms_dbfs, peak_dbfs (silence: null dBFS); not LUFS or true peak.
spectrum: Hann-windowed mono average, magnitudes and frequency_step in Hz; window contains 4..65536 power-of-two samples at 48000 Hz.
