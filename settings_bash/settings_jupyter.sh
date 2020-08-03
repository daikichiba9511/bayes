#!/bin/zsh

# ============= ここから jupyter の設定 ================
# --dev で開発用パッケージとして依存関係を固定する
poetry add --dev\
    jupyter \
    jupyterlab \
    jupytext \
    jupyter-contrib-nbextensions \
    jupyter-nbextensions-configurator \
    jupyterlab_code_formatter autopep8 black \
    jupyter-lsp

# jupyterの設定：jupytextの設定を生成したconfig.pyの末尾に追加
jupyter notebook --generate-config && \
    echo "\
c.ContentsManager.default_jupytext_formats = 'ipynb,py,jl'\n\
c.NotebookApp.contents_manager_class = 'jupytext.TextFileContentsManager'\n\
c.NotebookApp.open_browser = False\n\
" >> ${HOME}/.jupyter/jupyter_notebook_config.py

# jupyter notebook extension
jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user && \
    # enable extensions what you want
    jupyter nbextension enable select_keymap/main && \
    jupyter nbextension enable highlight_selected_word/main && \
    jupyter nbextension enable toggle_all_line_numbers/main && \
    jupyter nbextension enable varInspector/main && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable equation-numbering/main && \
    jupyter nbextension enable execute_time/ExecuteTime && \
    echo Done

# jupyter lab extension
jupyter labextension install @lckr/jupyterlab_variableinspector && \
    jupyter labextension install @jupyterlab/toc && \
    jupyter nbextension enable --py widgetsnbextension && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter labextension install @krassowski/jupyterlab-lsp
    jupyter labextension install @ryantam626/jupyterlab_code_formatter && \
    jupyter serverextension enable --py jupyterlab_code_formatter && \
    jupyter labextension install jupyterlab-theme-solarized-dark \
    # jupyter labextension install @z-m-k/jupyterlab_sublime && \
    # jupyter labextension install @hokyjack/jupyterlab-monokai-plus && \
    echo Done

# language server
pip3 install python-language-server[all]

# default の formatter : brack
mkdir -p /root/.jupyter/lab/user-settings/@ryantam626/jupyterlab_code_formatter && echo '\
{\n\
    "preferences": {\n\
        "default_formatter": {\n\
            "python": "black",\n\
        }\n\
    }\n\
}\n\
\
'>> /root/.jupyter/lab/user-settings/@ryantam626/jupyterlab_code_formatter/settings.jupyterlab-settings

# default で line number
mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/notebook-extension && echo '\
{\n\
    "codeCellConfig": {\n\
        "lineNumbers": true,\n\
    },\n\
}\n\
\
' >> /root/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings

echo " ========= settings all finished ==========="