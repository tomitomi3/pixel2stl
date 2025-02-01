# pixel2stl
pixel to STL converter / 画像をRGBの輝度情報を元にSTLを生成するコード。

1ピクセル毎に四角柱を作ることで実現する。大きい画像は処理が重くなりSTLのサイズも大きくなるので小さい画像向け。

## 必要なパッケージのインストール

> pip install trimesh==4.6.1 numpy opencv-python shapely manifold3d==3.0.1

## コマンド

> python pixel2stl.py [image_path] [cluster] [spacing] [z_height] [z_baseheight] [is_bright_z_thickness]

* cluster 減色数
* spascing STLにした時の1ピクセルの大きさ 単位はmm
* z_height STLにした時の高さ 単位はmm 
* z_baseheight ベース高さ 単位はmm
* is_bright_z_thickness 暗さを高さにする場合は0, 明るさを高さにする場合は1

### 例

> python pixel2stl.py sample.png 2 0.5 10 10 0

| Image | STL |
|:---:|:---:|
| <img src="https://github.com/tomitomi3/pixel2stl/blob/main/img/sample.png?raw=true" width="200"/> | <img src="https://github.com/tomitomi3/pixel2stl/blob/main/img/sample_to_stl.PNG?raw=true" width="200"/> |
