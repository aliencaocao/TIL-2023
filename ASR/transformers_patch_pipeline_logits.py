# All lines in Transformers v4.29.2
from collections import defaultdict

import numpy as np
import torch


# AutomaticSpeechRecognitionPipeline._forward transformers/pipelines/automatic_speech_recognition.py line 419
def _forward(self, model_inputs, return_timestamps=False, generate_kwargs=None):
    if generate_kwargs is None:
        generate_kwargs = {}
    if return_timestamps and self.type == "seq2seq_whisper":
        generate_kwargs["return_timestamps"] = return_timestamps
    is_last = model_inputs.pop("is_last")
    logits = None
    if self.type in {"seq2seq", "seq2seq_whisper"}:
        encoder = self.model.get_encoder()
        # Consume values so we can let extra information flow freely through
        # the pipeline (important for `partial` in microphone)
        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Seq2Seq speech recognition model requires either a "
                f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
            )

        # we need to pass `processed.get("attention_mask")` here since audio encoder
        # attention mask  length is different from expected text decoder `encoder_attention_mask` length
        # `generate` magic to create the mask automatically won't work, we basically need to help
        # it here.
        attention_mask = model_inputs.pop("attention_mask", None)
        tokens = self.model.generate(
            encoder_outputs=encoder(inputs, attention_mask=attention_mask),
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        out = {"tokens": tokens}
        if self.type == "seq2seq_whisper":
            stride = model_inputs.pop("stride", None)
            if stride is not None:
                out["stride"] = stride

    else:
        stride = model_inputs.pop("stride", None)
        input_values = model_inputs.pop("input_values")
        attention_mask = model_inputs.pop("attention_mask", None)
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        logits = outputs.logits

        if self.type == "ctc_with_lm":
            out = {"logits": logits}
        else:
            out = {"tokens": logits.argmax(dim=-1)}
        if stride is not None:
            # Send stride to `postprocess`.
            # it needs to be handled there where
            # the pieces are to be concatenated.
            ratio = 1 / self.model.config.inputs_to_logits_ratio
            if isinstance(stride, tuple):
                out["stride"] = rescale_stride([stride], ratio)[0]
            else:
                out["stride"] = rescale_stride(stride, ratio)
    # Leftover
    extra = model_inputs
    return {"is_last": is_last, **out, **extra}, logits


# AutomaticSpeechRecognitionPipeline.postprocess transformers/pipelines/automatic_speech_recognition.py line 480
def postprocess(self, model_outputs, decoder_kwargs= None, return_timestamps=None, return_language=None):
    # Optional return types
    optional = {}

    if return_timestamps and self.type == "seq2seq":
        raise ValueError("We cannot return_timestamps yet on non-ctc models apart from Whisper !")
    if return_timestamps == "char" and self.type == "ctc_with_lm":
        raise ValueError("CTC with LM cannot return `char` timestamps, only `words`")
    if return_timestamps in {"char", "words"} and self.type == "seq2seq_whisper":
        raise ValueError("Whisper cannot return `char` nor `words` timestamps, use `True` instead.")

    if return_language is not None and self.type != "seq2seq_whisper":
        raise ValueError("Only whisper can return language for now.")

    final_items = []
    key = "logits" if self.type == "ctc_with_lm" else "tokens"
    stride = None
    for outputs in model_outputs:
        outputs, logits = outputs
        items = outputs[key].numpy()
        stride = outputs.get("stride", None)
        if stride is not None and self.type in {"ctc", "ctc_with_lm"}:
            total_n, left, right = stride
            # Total_n might be < logits.shape[1]
            # because of padding, that's why
            # we need to reconstruct this information
            # This won't work with left padding (which doesn't exist right now)
            right_n = total_n - right
            items = items[:, left:right_n]
        final_items.append(items)

    if stride and self.type == "seq2seq":
        items = _find_longest_common_sequence(final_items, self.tokenizer)
    elif self.type == "seq2seq_whisper":
        time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions
        # Send the chunking back to seconds, it's easier to handle in whisper
        sampling_rate = self.feature_extractor.sampling_rate
        for output in model_outputs:
            if "stride" in output:
                chunk_len, stride_left, stride_right = output["stride"]
                # Go back in seconds
                chunk_len /= sampling_rate
                stride_left /= sampling_rate
                stride_right /= sampling_rate
                output["stride"] = chunk_len, stride_left, stride_right

        text, optional = self.tokenizer._decode_asr(
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )
    else:
        items = np.concatenate(final_items, axis=1)
        items = items.squeeze(0)

    if self.type == "ctc_with_lm":
        if decoder_kwargs is None:
            decoder_kwargs = {}
        beams = self.decoder.decode_beams(items, **decoder_kwargs)
        text = beams[0][0]
        if return_timestamps:
            # Simply cast from pyctcdecode format to wav2vec2 format to leverage
            # pre-existing code later
            chunk_offset = beams[0][2]
            offsets = []
            for word, (start_offset, end_offset) in chunk_offset:
                offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
    elif self.type != "seq2seq_whisper":
        skip_special_tokens = self.type != "ctc"
        text = self.tokenizer.decode(items, skip_special_tokens=skip_special_tokens)
        if return_timestamps:
            offsets = self.tokenizer.decode(
                items, skip_special_tokens=skip_special_tokens, output_char_offsets=True
            )["char_offsets"]
            if return_timestamps == "word":
                offsets = self.tokenizer._get_word_offsets(offsets, self.tokenizer.replace_word_delimiter_char)

    if return_timestamps and self.type not in {"seq2seq", "seq2seq_whisper"}:
        chunks = []
        for item in offsets:
            start = item["start_offset"] * self.model.config.inputs_to_logits_ratio
            start /= self.feature_extractor.sampling_rate

            stop = item["end_offset"] * self.model.config.inputs_to_logits_ratio
            stop /= self.feature_extractor.sampling_rate

            chunks.append({"text": item[return_timestamps], "timestamp": (start, stop)})
        optional["chunks"] = chunks

    extra = defaultdict(list)
    for output in model_outputs:
        output, logits = output
        output.pop("tokens", None)
        output.pop("logits", None)
        output.pop("is_last", None)
        output.pop("stride", None)
        for k, v in output.items():
            extra[k].append(v)
    return {"text": text, 'logits': logits, **optional, **extra}


# Pipeline.forward transformers\pipelines\base.py line 1016
def forward(self, model_inputs, **forward_params):
    with self.device_placement():
        if self.framework == "tf":
            model_inputs["training"] = False
            model_outputs = self._forward(model_inputs, **forward_params)
        elif self.framework == "pt":
            inference_context = self.get_inference_context()
            with inference_context():
                model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                model_outputs, logits = self._forward(model_inputs, **forward_params)
                model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
        else:
            raise ValueError(f"Framework {self.framework} is not supported")
    return model_outputs, logits


# ChunkPipeline.run_single transformers\pipelines\base.py line 1138
def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
    all_outputs = []
    for model_inputs in self.preprocess(inputs, **preprocess_params):
        model_outputs, logits = self.forward(model_inputs, **forward_params)
        all_outputs.append(model_outputs)
    outputs = self.postprocess(all_outputs, **postprocess_params)
    return outputs, logits


# PipelinePackIterator.__next__ transformers\pipelines\pt_utils.py line 246
def __next__(self):
        # Extremely similar to PipelineIterator in its unpacking mechanism
        # BUT, we have an extra required item which is the presence of `is_last`
        # That is because everything is flattened by `PipelineChunkIterator` we
        # need to keep track of how to regroup here in the original `process`
        # boundaries so that `process` and `postprocess` see the same data.

        # This iterator accumulates items (possibly while unbatching) until it
        # its a `is_last` and then just passes it on to the caller.
        is_last = False
        accumulator = []
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            while self._loader_batch_index < self.loader_batch_size:
                item = self.loader_batch_item()
                is_last = item.pop("is_last")
                accumulator.append(item)
                if is_last:
                    return accumulator

        while not is_last:
            processed = self.infer(next(self.iterator), **self.params)
            processed, logit = processed
            if self.loader_batch_size is not None:
                if isinstance(processed, torch.Tensor):
                    first_tensor = processed
                else:
                    key = list(processed.keys())[0]
                    first_tensor = processed[key]
                if isinstance(first_tensor, list):
                    observed_batch_size = len(first_tensor)
                else:
                    observed_batch_size = first_tensor.shape[0]
                if 0 < observed_batch_size < self.loader_batch_size:
                    # could be last batch so we can't unroll as many
                    # elements.
                    self.loader_batch_size = observed_batch_size
                self._loader_batch_data = processed
                self._loader_batch_index = 0
                while self._loader_batch_index < self.loader_batch_size:
                    item = self.loader_batch_item()
                    is_last = item.pop("is_last")
                    accumulator.append([item, logit[self._loader_batch_index]])
                    if is_last:
                        return accumulator
            else:
                item = processed
                is_last = item.pop("is_last")
                accumulator.append([item, logit])
        return accumulator