#!/bin/zsh
# =========== 使い方 ===================
# poetry は入っている前提
# `$ chmod 755 create_python_env.sh` で権限を与える
# `$ ./settings_python_env.sh'

# ========== poetry ============
# project directory 下に'.venv'を作る
poetry config virtualenvs.in-project true
# 補完
poetry completions zsh > ~/.zfunc/_poetry

# 変更の適用
source ~/.zshrc
