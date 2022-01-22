# README

## What's this

this reposiotry has codes which I learned about bayesian statistics.

So, If you find some mistakes, let me know by creating Issue. I would like to fix about it or do prompt response as much as possible.

If you want to know more aobut baysian statistics, I recomends that you should visit Reference[5] that some contents about bayesian statistics, learning theory  wrote by Japanese, but there are English ver.

### stan_with_python

If you want to know about this, Please see Reference[2]

but Reference[2] is wrote by Japanese...

I have wrote this codes by python while reffering to Ref[2]

Some note about this directory bellow.

* there are some dataset which is used in Ref[2], are under author repository
  * If you needs, Please see this repository: (https://github.com/MatsuuraKentaro/RStanBook)

* I used pystan(Stan) to do probababilistic modeling.

  * details of Installation -> (https://pystan.readthedocs.io/en/latest/installation_beginner.html

  * Stan needs g++(C++ compiler), so If you dont have g++ or dont set enviromentvariables as c++ compiler you shuld do it.

* I use `arviz` to make some plot like posterior distribution which is sampled by pystan
    * details -> (https://arviz-devs.github.io/arviz/examples/index.html)

### stan_with julia

I rewrote by Julia about `stan_with_python`

### watanabe_bayes

some codes about Reference[1] but, still under progressiong

## Reference

* [1]  [ベイズ統計の理論と方法 渡辺澄夫](https://www.amazon.co.jp/%E3%83%99%E3%82%A4%E3%82%BA%E7%B5%B1%E8%A8%88%E3%81%AE%E7%90%86%E8%AB%96%E3%81%A8%E6%96%B9%E6%B3%95-%E6%B8%A1%E8%BE%BA-%E6%BE%84%E5%A4%AB/dp/4339024627/ref=sr_1_1?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&keywords=%E6%B8%A1%E8%BE%BA%E6%BE%84%E5%A4%AB&qid=1583001040&sr=8-1)
* [2] [StanとRでベイズ統計モデリング(Wonderful R) 　松浦健太郎](https://www.amazon.co.jp/Stan%E3%81%A8R%E3%81%A7%E3%83%99%E3%82%A4%E3%82%BA%E7%B5%B1%E8%A8%88%E3%83%A2%E3%83%87%E3%83%AA%E3%83%B3%E3%82%B0-Wonderful-R-%E6%9D%BE%E6%B5%A6-%E5%81%A5%E5%A4%AA%E9%83%8E/dp/4320112423/ref=sr_1_9?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&keywords=%E3%83%99%E3%82%A4%E3%82%BA%E7%B5%B1%E8%A8%88&qid=1583001477&sr=8-9)
* [3] [機械学習スタートアップシリーズ ベイズ推論による機械学習入門 (KS情報科学専門書) 須山 敦志](https://www.amazon.co.jp/%E9%A0%88%E5%B1%B1-%E6%95%A6%E5%BF%97/e/B078JW6FN2/ref=dp_byline_cont_book_1)
* [4] [JuliaでStan ~ 環境構築編 ~](https://blog.hatena.ne.jp/daiki_tech/daiki-tech.hatenablog.com/edit?entry=26006613609655045)

* [5] [渡辺澄夫, 渡辺澄夫 (2022/01/22)](http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/index-j.html)