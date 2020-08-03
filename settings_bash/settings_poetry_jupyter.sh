#!/bin/zsh

# ============= ここから jupyter の設定 ================
# --dev で開発用パッケージとして依存関係を固定する
poetry add --dev jupyter jupytext jupyter-contrib-nbextensions jupyter-nbextensions-configurator \
    jupyterlab_code_formatter autopep8 black jupyter-lsp

# jupyterの設定：jupytextの設定を生成したconfig.pyの末尾に追加
poetry run jupyter notebook --generate-config
echo "\
c.ContentsManager.default_jupytext_formats = 'ipynb,py,jl'\n\
c.NotebookApp.contents_manager_class = 'jupytext.TextFileContentsManager'\n\
c.NotebookApp.open_browser = False\n\
" >> ${PWD}/.venv/.jupyter/jupyter_notebook_config.py

# jupyter notebook extension
poetry run jupyter contrib nbextension install --user
poetry run jupyter nbextensions_configurator enable --user
# enable extensions what you want
poetry run jupyter nbextension enable select_keymap/main
poetry run jupyter nbextension enable highlight_selected_word/main
poetry run jupyter nbextension enable toggle_all_line_numbers/main
poetry run jupyter nbextension enable varInspector/main
poetry run jupyter nbextension enable toc2/main
poetry run jupyter nbextension enable equation-numbering/main
poetry run jupyter nbextension enable execute_time/ExecuteTime
echo Done

# jupyter lab extension
poetry run jupyter labextension install @lckr/jupyterlab_variableinspector
poetry run jupyter labextension install @jupyterlab/toc
poetry run jupyter nbextension enable --py widgetsnbextension
poetry run jupyter labextension install @jupyter-widgets/jupyterlab-manager
poetry run jupyter labextension install @krassowski/jupyterlab-lsp
poetry run jupyter labextension install @ryantam626/jupyterlab_code_formatter
poetry run jupyter serverextension enable --py jupyterlab_code_formatter
poetry run jupyter labextension install jupyterlab-theme-solarized-dark
# jupyter labextension install @z-m-k/jupyterlab_sublime && \
# jupyter labextension install @hokyjack/jupyterlab-monokai-plus && \
echo Done

# language server
pip3 install "python-language-server[all]"

# default の formatter : brack
mkdir -p ${PWD}/.venv/.jupyter/lab/user-settings/@ryantam626/jupyterlab_code_formatter

echo '\
{\n\
    "preferences": {\n\
        "default_formatter": {\n\
            "python": "black",\n\
        }\n\
    }\n\
}\n\
\
'>> ${PWD}/.venv/.jupyter/lab/user-settings/@ryantam626/jupyterlab_code_formatter/settings.jupyterlab-settings

# default で line number
mkdir -p ${PWD}/.venv/.jupyter/lab/user-settings/@jupyterlab/notebook-extension

echo '\
{\n\
    "codeCellConfig": {\n\
        "lineNumbers": true,\n\
    },\n\
}\n\
\
' >> ${PWD}/.venv/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings

echo " ========= settings all finished ==========="