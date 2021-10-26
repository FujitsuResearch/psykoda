# 開発方針
* 困り driven で柔軟に変更していく
* できるだけ変更の根拠を残す
* 自明になった項目は削除する

## リポジトリ管理
* master に直接 commit & push せず Pull Request 経由で変更する
    * merge には approve を要する設定にしてある
    * タイトルに `[WIP]` (work in progress), `[FAILING]` などのあるものは merge しない
    * review request は `[WIP]` (など) を取り除いたタイミングで行う (request される前に見てコメントするのはよいが review はしない)
    * そろそろ CI が欲しい

## (まだ) 利用し (てい) ないツール
* `pytest-cov`: `coverage`+`pytest` では不便な場合

## [Python versions](pythons-ja.md)
* 3.5以下: 不要 (OSS化プロジェクト開始時点 (Day 0) で end-of-life)
* 3.6, 3.7: 依存関係の問題で対象外
* 3.8: オリジナル版 requirement (3.8.5)
* 3.9: Day 0 の最新リリース (3.9.2, hotfixes: 3.9.4)
* 3.10: 未 ([expected: 2021/10](https://www.python.org/dev/peps/pep-0619/))
