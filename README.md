# BertSum
Tensorflow Implementation of "Text Summarization with Pretrained Encoders", BertSum model with Korean and tf.RaggedTensor.

- Dataset: <a href="https://aihub.or.kr/aidata/8054" target="_blank"> AI 허브 문서요약 텍스트 데이터 </a>
- 불용어 인덱스의 명확한 의미를 인지하지 못하여 불용어 처리를 하지 않음.

1. Extractive BertSum

- 문서 안에서 요약 문장 추출 모델 -> 문장 분류 모델

2. Abstractive BertSum (추후 추가 예정)

- 문서 요약문 생성 모델 -> 문장 생성 모델
