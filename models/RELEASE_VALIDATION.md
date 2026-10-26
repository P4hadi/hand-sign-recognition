# Final Classifier Release Validation

- Source artifact was loaded from the original project without modification.
- Sanitization changed only top-level `training_datasets` metadata paths.
- Original and sanitized estimators were compared on 32 deterministic synthetic 63-feature inputs.
- Predicted labels matched for all inputs.
- Probability matrices matched exactly for all inputs.
- Labels: `open_palm`, `fist`, `thumbs_up`, `peace_sign`, `ok_sign`, `pointing_up`, `three_fingers`, `call_me`.
- Feature count: 63.
- Probability support: yes.
- Release size: 4159530 bytes.
- SHA-256: `57292904bd781a1b0d0d6867d7f9e81727dcf69cff72abcfe4818931fe0810b1`.
