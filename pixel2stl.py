from stl import mesh
import trimesh
import numpy as np
import mapbox_earcut as earcut
import cv2
import os
import sys

# 必要なパッケージ
# pip install numpy-stl trimesh numpy mapbox-earcut opencv-python


def Gensyoku(imgpath, cluster=8):
    # K-Meansクラスタリングによる画像の減色
    # https://tat-pytone.hatenablog.com/entry/2019/06/13/210251

    # 画像の読み出し
    img = cv2.imread(imgpath)

    # ndarray(y,x,[B,G,R])を変形(y * x,[B,G,R])
    Z = img.reshape((-1, 3))

    # float32に型変換
    Z = np.float32(Z)

    # 計算終了条件の設定。指定された精度(1.0)か指定された回数(10)計算したら終了
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # クラスター数
    K = cluster
    # K-Meansクラスタリングの実施
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 各クラスタの中心色リストcenterをunit8に型変換
    center = np.uint8(center)
    # 中心色リストcenterから分類結果labelの要素を取り出す
    res = center[label.flatten()]
    # 元画像の形状に変形
    res2 = res.reshape((img.shape))

    return res2


def ExtrudeZDirection(polygonPoints, pixelValueArray,
                      thickness, base_height=0.0):
    # 全てのポリゴンに対して処理を行う
    trimesh_meshes = []
    for i, points in enumerate(polygonPoints):
        # 1000ポリゴンごとに処理状況を出力
        if i % 1000 == 0:
            print(f"{i} polygons processed...")

        # thicknessをかけてZ方向の高さとしbase_heightを加算
        pixel_height = base_height + thickness * pixelValueArray[i]

        # pixel_height が0の場合は次へ
        if pixel_height == 0:
            continue

        # 頂点情報の取得
        num_vertices = len(points)
        vertices = np.zeros((num_vertices * 2, 3))
        for j in range(num_vertices):
            vertices[j, :2] = points[j]
            vertices[j, 2] = 0
            vertices[num_vertices + j, :2] = points[j]
            vertices[num_vertices + j, 2] = pixel_height

        # 頂点情報を組み合わせた座標データ
        verts = np.array(points)

        # 非凸ポリゴンを三角形に分割するけど、普通のtriangulationでもよい気がする
        rings = np.array([len(verts)])
        triangles = earcut.triangulate_float32(verts, rings)

        # 分割三角毎にreshape
        triangles = triangles.reshape(-1, 3)

        # 上面と下面の面を定義
        top_faces = []
        bottom_faces = []
        for triangle in triangles.tolist():
            top_faces.append([triangle[0], triangle[1], triangle[2]])
            bottom_faces.append(
                [triangle[0] + num_vertices, triangle[1] + num_vertices, triangle[2] + num_vertices])

        # 側面を定義
        side_faces = []
        for k in range(num_vertices):
            next_k = (k + 1) % num_vertices
            side_faces.append([k, next_k, k + num_vertices])
            side_faces.append(
                [next_k, next_k + num_vertices, k + num_vertices])

        # 面を結合
        faces = np.array(top_faces + bottom_faces + side_faces)

        # メッシュにする
        polygon_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for l, f in enumerate(faces):
            for m in range(3):
                polygon_mesh.vectors[l][m] = vertices[f[m], :]

        # Trimeshメッシュに変換
        vertices = polygon_mesh.vectors.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 修正
        # trimesh.repair.fix_normals(trimesh_mesh)

        trimesh_meshes.append(trimesh_mesh)

    # 複数の Trimesh メッシュが生成されるため、1つのメッシュに結合する
    concatTriMesh = trimesh.util.concatenate(trimesh_meshes)

    # 処理完了を表示
    print("complete.")

    return concatTriMesh


if __name__ == "__main__":
    # デフォルト値の設定
    default_cluster = 8         # 減色数
    default_spacing = 0.5       # 1ピクセルの大きさ mm
    default_z_height = 1.0      # Zの高さ mm
    default_base_height = 1.0   # ベース長さ mm
    # True:輝度が高いとZ方向に厚み, False:輝度が低いとZ方向に厚み
    default_is_bright_z_thickness = True

    # 引数から画像パスを取得
    img_path = ""
    if len(sys.argv) != 7:
        print("Usage:")
        print(
            ">python pixel2stl.py [image_path] [cluster] [spacing] [z_height] [z_baseheight] [is_bright_z_thickness]")
        print("ex:")
        print(">python pixel2stl.py sample.png 8 0.5 10.0 10.0 1")
        sys.exit(1)
    else:
        img_path = sys.argv[1]

    # 入力ファイルの存在チェック
    if not os.path.exists(img_path):
        print(f"{img_path} is not exist.")
        sys.exit(1)

    # 追加引数が提供されているかどうかに応じて変数を設定 sys.argv[1]は画像パス
    cluster = int(sys.argv[2]) if len(sys.argv) > 2 else default_cluster
    spacing = float(sys.argv[3]) if len(sys.argv) > 3 else default_spacing
    z_height = float(sys.argv[4]) if len(sys.argv) > 4 else default_z_height
    z_baseheight = float(sys.argv[5]) if len(sys.argv) > 5 else default_base_height
    is_bright_z_thickness = bool(int(sys.argv[6])) if len(sys.argv) > 6 else default_is_bright_z_thickness

    print(f"Input image path: {img_path}")
    print(f"Cluster: {cluster}, Spacing: {spacing}, Z height: {z_height}, Z base height: {z_baseheight}, is_bright_z_thickness: {is_bright_z_thickness}")

    # 減色してグレースケール画像を入手する
    print(f"input image path:{img_path}")
    gensyoku_img = Gensyoku(img_path, cluster)
    grayscale_img = cv2.cvtColor(gensyoku_img, cv2.COLOR_BGR2GRAY)
    grayscaleArray = np.array(grayscale_img)

    # 0-1正規化
    scale = grayscaleArray.max() - grayscaleArray.min()
    grayscaleArray = (grayscaleArray - grayscaleArray.min()) / scale

    # 輝度情報を元にZ方向反転
    if is_bright_z_thickness == False:
        grayscaleArray = (grayscaleArray * -1.0) + 1.0

    # 1ピクセルを輝度情報を元にZ方向に押し出す
    rectPolygons = []
    tempZValues = []
    for x in range(grayscaleArray.shape[0]):
        for y in range(grayscaleArray.shape[1]):
            # 四角形を作る
            top_left_x = x * spacing
            top_left_y = y * spacing

            points = [
                (top_left_x, top_left_y),
                (top_left_x, top_left_y + spacing),
                (top_left_x + spacing, top_left_y + spacing),
                (top_left_x + spacing, top_left_y)
            ]

            rectPolygons.append(points)
            tempZValues.append(grayscaleArray[x, y])

    zValues = np.array(tempZValues)

    # Z方向で押し出し
    combined_mesh = ExtrudeZDirection(
        rectPolygons, zValues, z_height, z_baseheight)

    # STL出力
    filename_without_ext = os.path.splitext(os.path.basename(img_path))[0]
    stl_filename = f"{filename_without_ext}.stl"
    print(f"ouput stl path:{stl_filename}")

    combined_mesh.export(stl_filename)
