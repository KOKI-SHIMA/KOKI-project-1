# AnimeTwin

AnimeTwinは、撮影した人物の顔と見た目が似ているアニメキャラクターを、AIを使って探すプログラムです。

このプロジェクトは、NVIDIA Jetson Orin Nano上で動作します。

## 仕組み

1. USB cameraで写真を撮影します。
2. OpenCV YuNetを使って顔を検出し、顔の部分を切り取ります。
3. OpenAI CLIPを使って、顔画像の視覚的な特徴を数値に変換します。
4. 人物の顔と、localに保存されたキャラクター画像を比較します。
5. 類似度が高い上位3人をHTMLの結果画面に表示します。

## 使用技術

- NVIDIA Jetson Orin Nano
- Python
- PyTorch
- CUDA
- OpenCV
- YuNet
- OpenAI CLIP
- Docker

## File構成

- `src/capture.py`：USB cameraで写真を撮影します。
- `src/detect_face.py`：写真から顔を検出して切り取ります。
- `src/compare_clip.py`：CLIPを使って画像の特徴を比較します。
- `src/create_result_page.py`：比較結果をHTMLに変換します。
- `src/main.py`：すべての処理を順番に実行します。
- `requirements.txt`：追加で必要なPython packageを記録しています。

## 実行方法

JetsonのDocker container内で、次のcommandを実行します。

```bash
cd /workspace/anime-twin
python3 src/main.py
```

結果画面を公開するために、次のcommandを実行します。

```bash
python3 -m http.server 8000 --directory /workspace/anime-twin/output
```

同じnetworkに接続したPCのbrowserで、次のURLを開きます。

```text
http://JETSON_IP_ADDRESS:8000/result.html
```

現在の環境では、次のURLを使用します。

```text
http://192.168.137.199:8000/result.html
```

## 出力されるFile

- `output/face_crop.jpg`：切り取られた顔画像
- `output/face_detected.jpg`：顔の検出位置を示した画像
- `output/clip_results.csv`：キャラクターとの類似度
- `output/result.html`：最終結果画面

## 注意事項

- Cosine similarityは画像同士の特徴の近さを表す数値であり、確率ではありません。
- このプログラムは「顔が完全に同じ」と判断するものではありません。
- キャラクター画像、撮影画像、AI model、生成された結果はGitHubに公開しません。
- 使用する画像について、著作権・肖像権・privacyに注意します。
- 人物写真を使用する場合は、本人の許可を得ます。

## 現在の状態

Prototype完成。