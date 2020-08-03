#!/bin/zsh
# =========== 使い方 ===================
# git は入っている前提
# `$ chmod 755 create_python_env.sh` で権限を与える
# `$ ./create_python_env.sh'

# ================ pyenv ================
# pyenv のインストール
# brew install でも問題ないがpathがusr/local/に入る
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# pyenvのPATHを通す
echo '# ===============================' >> ~/.zshrc
echo '# pyenv' >> ~/.zshrc
echo '# ===============================' >> ~/.zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.zshrc
# 変更の適用
source ~/.zshrc

# ============ python ==============
# python の install
echo "===== python install start ====="
pyenv install 3.7.6
# 変更の適用
source ~/.zshrc
echo "===== python install finish ====="

# ========== poetry ============
# poetry の install
# openssl の installで時間がかかるかも。。
echo "======= poetry install start ========"
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
# 変更の適用
source ~/.zshrc
echo "======= poetry install finish ========"

# 終わったらzshを再起動した方が良いかも。。