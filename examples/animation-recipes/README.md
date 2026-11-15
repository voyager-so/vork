# Animation recipes

Run from the checkout root; Python 3.10+ and vork are required. No Python packages.
Generators write JSON to stdout; the caption example resolves its bundled font
relative to its script, so the JSON can live outside the checkout.

```sh
python3 examples/animation-recipes/spring.py > /tmp/spring.json
python3 examples/animation-recipes/sequences.py > /tmp/sequences.json
python3 examples/animation-recipes/captions.py examples/animation-recipes/captions.srt > /tmp/captions.json
vork render /tmp/spring.json --out /tmp/spring.mp4
vork render /tmp/sequences.json --out /tmp/sequences.mp4
vork render /tmp/captions.json --out /tmp/captions.mp4
```

`vork help springs`, `vork help sequences` and `vork help captions` describe the
track and clock contracts. `spring.py` draws a gold spring and a blue smoothstep;
`sequences.py` dissolves blue into pink; `captions.py` keeps SRT line breaks.
SRT markup is literal text; overlapping cues, positioning and transcription are
outside this example. Edit the text node for typography and placement.
