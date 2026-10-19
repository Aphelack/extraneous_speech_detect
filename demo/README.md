# Demo records

Generated with held-out Russian speakers `ru_spk_5..7`, seed-controlled and 7 seconds per record.

Each case contains five records:

- `normal/` - ordinary conversation with acoustic leakage;
- `double_talk/` - overlapping speech;
- `high_leak_lt_10db/` - strong leakage below 10 dB attenuation;
- `path_drift/` - changing leakage delay;
- `third_speaker/` - an additional speaker near channel A.

Every `record_XX/` contains:

- `input.wav` - synthesized stereo input;
- `ground_truth.json` - generated interval labels;
- `predictions.json` - output of `detect.py`;
- `cleaned.wav` - output of `clean.py`;
- `metadata.json` - scenario, seed, speakers, and synthesis parameters.

To reproduce one record manually:

```bash
python detect.py demo/third_speaker/record_01/input.wav /tmp/predictions.json
python clean.py demo/third_speaker/record_01/input.wav /tmp/cleaned.wav
```
