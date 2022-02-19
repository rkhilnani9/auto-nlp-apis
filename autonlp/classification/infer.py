import re
from transformers import pipeline
from autonlp.classification.model import get_model, get_tokenizer

model = get_model()
tokenizer = get_tokenizer()


def infer(input_text, pretrained):
    model = get_model()
    tokenizer = get_tokenizer()
    input_text = re.sub(r"[^\w\s]", "", input_text).lower()

    if pretrained == "true":
        sentiment_classifier = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer
        )
        preds = sentiment_classifier(input_text)[0]
    else:
        sentiment_classifier = AutoModelForSequenceClassification.from_pretrained(config.CLS_MODEL_SAVE_PATH + "checkpoint-500/")
        tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)
        tokens = tokenizer(input_text)
        logits = sentiment_classifier(**tokens)
        pred_prob = np.max(torch.sigmoid(logits["logits"]).detach().cpu().numpy())
        label = "POSITIVE" if pred_prob > 0.5 else "NEGATIVE"
        preds = {"label" : label, "score" : pred_prob}


    return preds


