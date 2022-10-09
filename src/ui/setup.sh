mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

python -m spacy download en_core_web_sm

DRIVE_URL="https://drive.google.com/uc?id="
MODEL_ID=""1AvLBaLLdT8MeCQNbSx2FxiT5y7cDtVsE
gdown ${DRIVE_URL}${MODEL_ID}
