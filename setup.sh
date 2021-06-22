mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"akhila9040@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
ehco "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port=$PORT\n\
" >~/.streamlit/config.toml
