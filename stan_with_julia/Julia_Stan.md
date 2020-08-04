# StanをJuliaで使っていく。

メインは参考資料１をJuliaとJuliaStanを使って書いていく。

## 準備

juliaのインストールは他の人がわかりやすく書いてるので、他の記事を参照。

stan.jlを使おうと思ったけど色々あって、PyCall/pystanに切り替え

環境変数にいれるのに、仮想環境(自分場合はminiconda)のPythonも使えた。

minicondaのすでにある仮想環境を使いたい時、

Conda.ROOTENVで確認。自分の使いたい環境でなければ

`ENV["CONDA_JL_HOME"] = "使いたい環境までのpath" `

これで、使いたい環境を切り替える。

- `Pkg> bulid Conda` でrebulid

- `PyCall["PYTHON"]="使いたいPythonへのPath"`

- `PyCall.pyversion`とか`PyCall.libpyhton`でバージョン確認できる

一旦ターミナルを再起動した方がいいかもしれない。。

pythonまでのpathは、使いたい環境で`which python3`

詳しくは　-> [もっと早く見つけたかった。。](https://qiita.com/ysaito8015@github/items/bee0846c227b10f3f369)

juliaからPyCall経由でpystan使ってみたけど、遅すぎ？なのかうまく使えてないから、stan.jlをやるか、pythonでpystanを先にやる方がいい。

#### 2020/08/03環境構築について更新

JuliaでStan.jlを使うための備忘録を書きました。詳細は以下リンク先を参照ください。

* [Julia で Stan ~ 環境構築編 ~](https://blog.hatena.ne.jp/daiki_tech/daiki-tech.hatenablog.com/edit?entry=26006613609655045)

### version

- Julia 1.5


## 参考資料

1. [StanとRでベイズ東経モデリング  松浦健太郎]()
