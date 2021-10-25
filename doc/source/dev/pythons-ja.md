# Python versions
Python のインストールは `tox` が自動で行うのではなく、あらかじめ行っておいて `tox` に知らせる必要がある。

## Windows
* [環境変数 `TOX_DISCOVER`](https://tox.readthedocs.io/en/latest/config.html#environment-variables) を [`tox.interpreters.common:base_discover` が解釈する](https://github.com/tox-dev/tox/blob/7aa130318d168ccc476d856ff7b1531bdaf263cc/src/tox/interpreters/common.py#L14)
* [`tox` のコマンドラインオプションで `--discover C:\..\python.exe`](https://tox.readthedocs.io/en/latest/config.html#cmdoption-tox-discover) と **フルパス** を指定 (複数可) することができる
* 環境を `C:\Python38` などに作成すると [`tox.interpreters.windows:tox_get_python_executable` が自動で見つける](https://github.com/tox-dev/tox/blob/master/src/tox/interpreters/windows/__init__.py)
