# img2stl
Image 2 STL converter / 画像からSTLを生成するコード

1ピクセル毎に四角柱を作ることで実現する。大きい画像は処理が重くなるので小さい画像向け。

## 必要なパッケージのインストール

> pip install numpy-stl trimesh numpy mapbox-earcut opencv-python 

## コマンド

>python thisscript.py image_path [cluster] [spacing] [z_height]

* cluster 減色数
* spascing X,Y座標における1ボクセルの大きさ 単位はmm
* z_height 高さ 単位はmm 

### 例

> python img2stl.py sample.png 8 0.5 10
