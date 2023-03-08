


仮想環境作構築
仮想環境を作成し、有効化
python -m venv myenv
./venv/Scripts/activate (Windows)
source myenv/bin/activate (Mac)

必要なライブラリのインストールｔ
pip install -r requirements.txt


ディレクトリ構成

CoachingSystem
├── CoachingSystem.py
├── ReadMe.txt
├── csv
│   ├── copycat　　　　　　　　　<- 手本者の骨格座標フォルダ
│   └── user　　　　　　　　　　　<- 利用者の骨格座標フォルダ
├── data
│   ├── edit_movie　　　　　　　　<- 編集後の動画格納フォルダ
│   ├── excel
│   ├── images　　　　　　　　　　<- 画像格納フォルダ
│   ├── movie　　　　　　　　　　　<- 利用者の動画入力フォルダ
│   ├── original_movie　　　　　<- 元の動画データ格納フォルダ
│   └── time_crop
├── editvideo.py            　<- 動画編集用スクリプト
├── experiments.ipynb
├── function.py　　　　　　　　　<- 作成したモジュールを格納
├── human_pose_estimation.py　<- 骨格推定用スクリプト
├── main.py　　　　　　　　　　　<- 実行スクリプト
└── requirements.txt　　　　　　<- 環境構築用ファイル

$ python main.py [利用者のファイル名] [手本者のファイル名] 閾値　\
　X1 Y1 X2 Y2

ex.
$ python main.py user copy 0.5 \
450 100 750 800