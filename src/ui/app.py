from src.ui.io import load_definitions, load_model, load_pretrained_tokenizer, load_spacy_tokenizer
from src.ui.utilities import preprocess_sentence, predict, get_disambiguated_tokens
from src.readers.wordnet_reader import WordNetReader
from src.libs.annotated_text import annotated_text
from src.utils.utilities import collate_fn

from omegaconf import DictConfig
import streamlit as st
import hydra


@hydra.main(config_path="../../conf", config_name="root")
def main(conf: DictConfig) -> None:
    # Configuration variables
    st.set_page_config(page_title="Neural WSD")
    sentence = "How do you assess the quality of your word embeddings?"

    # Additional required resources
    output_vocab = WordNetReader.vocabulary(conf)
    lexeme_means = WordNetReader.lexeme_means(conf)
    glosses = load_definitions(conf)

    # Load pre-trained model for inference
    model = load_model(conf, len(output_vocab))

    # Load word-level SpaCy tokenizer and model pretrained tokenizer
    bert_tokenizer = load_pretrained_tokenizer(conf.model.tokenizer)
    word_tokenizer = load_spacy_tokenizer()

    # general configuration variables
    colors = ["#EC7063", "#7FB3D5", "#73C6B6", "#F4D03F", "#EB984E", "#AAB7B8", "#D7BDE2", "#C499D6", "#63CDBF"]
    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = 0

    # ====== LAYOUT SETUP
    thr = st.sidebar.slider("Disambiguation treshold", 0.0, 1.0, 0.5)
    st.title("Word Sense Disambiguation")
    sentence = st.text_input("Input text", sentence)
    button_clicked = st.button("Analyze")

    # ====== MODEL INFERENCE
    preprocessed_sentence = preprocess_sentence(sentence, bert_tokenizer, word_tokenizer)
    batch_keys = list(preprocessed_sentence.keys())
    batch = collate_fn([preprocessed_sentence], batch_keys, lexeme_means=lexeme_means, output_dim=len(output_vocab))
    output_batch = predict(model, batch)
    disambiguated_tokens = get_disambiguated_tokens(preprocessed_sentence, output_batch, output_vocab, thr)

    # check validity of selected idx
    if st.session_state.selected_idx >= len(disambiguated_tokens):
        st.session_state.selected_idx = 0

    # ====== SECTION POS TAGGING
    st.subheader("POS tagging")
    pos_tags = [(t.text, t.pos_, colors[t.pos % len(colors)]) for t in preprocessed_sentence["tokens"]]
    annotated_text(*pos_tags)

    # ====== SECTION PREDICTION ANALYSIS
    st.text("")
    st.subheader("Prediction analysis")
    col1, col2 = st.columns((1, 1))

    with col1:
        for idx, (_, token, score, wn_id) in enumerate(disambiguated_tokens):
            st.button(f"{token.text}", key=wn_id, on_click=update_selected_index, args=(idx,))

    with col2:
        token_idx, token, score, wn_id = disambiguated_tokens[st.session_state.selected_idx]
        st.markdown(f"**Synset:** wn:{wn_id}")
        st.markdown("**Probability**: {:.2f}%".format(round(100 * score, 4)))
        st.markdown(f"**Semantic ambiguity:** {batch['sense_mask'].count_nonzero(-1).squeeze()[token_idx]}")
        st.write(f"**Definition**: {glosses[wn_id]}")

    candidate_idxs = batch["sense_mask"].squeeze()[token_idx].nonzero()
    candidate_scores = output_batch["scores"][token_idx, batch["sense_mask"].squeeze()[token_idx]]
    sorted_idxs = candidate_scores.argsort(descending=True)
    candidate_idxs, candidate_scores = candidate_idxs[sorted_idxs], candidate_scores[sorted_idxs]
    st.bar_chart(
        [{f"wn:{output_vocab.itos[idx]}": score.item()} for idx, score in zip(candidate_idxs, candidate_scores)]
    )

    # ====== SECTION ADDITIONAL INFORMATION
    st.subheader("Tokenization information")
    st.table(
        {
            "Word": token.text,
            "Lemma": token.lemma_,
            "POS": token.pos_,
            "Subword tokens": bert_tokenizer.tokenize(token.text),
        }
        for token in preprocessed_sentence["tokens"]
    )
    return


def update_selected_index(idx) -> None:
    """Utility function to update selected index from session state."""
    st.session_state.selected_idx = idx


if __name__ == "__main__":
    main()
