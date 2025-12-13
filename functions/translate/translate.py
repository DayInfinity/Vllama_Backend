from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# Initializing translation model
def init_translation_model(model_id):
    model_name = model_id
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).half()
    model = torch.compile(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


# Fast translation function
def translate_fast(text: str = 'Hello world!', model_id: str = 'facebook/nllb-200-distilled-600M', input_lang="de", output_lang="en", max_chunk_chars=2000):

    model, tokenizer, device = init_translation_model(model_id)


    LANGUAGE_CODES = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "hi": "hin_Deva",
        "zh": "zho_Hans",
        "ar": "arb_Arab",
    }

    src = LANGUAGE_CODES[input_lang]
    tgt = LANGUAGE_CODES[output_lang]

    tokenizer.src_lang = src

    chunks = []
    words = text.split()
    temp = []
    length = 0

    for w in words:
        if length + len(w) > max_chunk_chars:
            chunks.append(" ".join(temp))
            temp = []
            length = 0
        temp.append(w)
        length += len(w)
    if temp:
        chunks.append(" ".join(temp))

    outputs = []
    for chunk in chunks:
        enc = tokenizer(chunk, return_tensors="pt", truncation=True).to(device)

        result = model.generate(
            **enc,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt),
            max_length=512,
            num_beams=1,  # FAST
        )

        outputs.append(tokenizer.decode(result[0], skip_special_tokens=True))

    output_text = " ".join(outputs)

    return output_text
