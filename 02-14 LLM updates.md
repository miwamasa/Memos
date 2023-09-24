---


---

<h1 id="ひたすらllm関連情報を追う、">ひたすらLLM関連情報を追う、</h1>
<p>これは、個人のtwitter bookmarkを毎週おさらいしている。</p>
<h2 id="section">9/25</h2>
<p>相も合わらず、RAG(Retrieval Augmented Generation)関係が多いのはご容赦。上位のLLM(GPT-4とか）をつかって正解をつくって、RAGを評価する仕組みとか、この評価の仕組みをつかって別のLLＭ(gpt-3.5-turboとか)をRAG向けにfine-tuningするなんてのが、e2e(end-to-end)の手法として当たり前になりつつある。「知識は樹木のようなもの」とのたまうスクエニの三宅さんの話はいつも面白い。SOPをつかったAgentsというのはagentの可制御性という意味で面白い。Transformers.jsをつかったWeb LLMの新手が登場。Xwin-LM-70BがGPT-4超えか？というのがもっぱらの話題。LLMが創造性を持つか？の論文での創造性の３つの基準（価値、新規性、驚き）って、特許提案と同じだよね、LLMが特許提案できるか？に置き換えても同じ。instructorというopenai function callingにpydanticを組み合わせられるライブラリ使ってみたい。RAGでもメタ情報抽出にpydantic使ったりとか、この辺りも定番化か。ChatGPTの知識が、2022年1月までの知識までアプデされた。LLMの利用サーベイ、「５位：ビジネス戦略立案」ってのは笑ったね。gpt-3.5-turbo-instructというのが出てるのね、コンパクトで、言語生成に適したモデル（チャット用ではない）、これはfine-tuning用なのか？？、LLM向けAI半導体「SN40L」ってのも期待。</p>
<ul>
<li>ちょっとした気配りで皆を幸せにする GitHub の使い方
<ul>
<li><a href="https://qiita.com/squid-cat/items/7166317e60d3ff96ccb7">https://qiita.com/squid-cat/items/7166317e60d3ff96ccb7</a></li>
<li>PR がレビューされない環境を作らない</li>
</ul>
</li>
<li>米国のAI企業公聴会より、Nvidiaの証言が素晴らしい
<ul>
<li><a href="https://x.com/Yampeleg/status/1703774531771363738?s=20">https://x.com/Yampeleg/status/1703774531771363738?s=20</a></li>
<li>OpenAI: AI will kill us.</li>
<li>Anthropic: AI will kill us.</li>
<li>InflectionAI: AI will kill us.</li>
<li>Nvidia: Fortunately uncontrollable Artificial General Intelligence is Science Fiction not reality.</li>
</ul>
</li>
<li>知識と技術の継承としてのAI by スクエニ三宅さん
<ul>
<li><a href="https://togetter.com/li/2226417">https://togetter.com/li/2226417</a></li>
<li>その分野の専門家が持つそういった知識体系が、その教授なり専門家の価値なわけであるが、実際のところ、近くにいて話しかけなければ、自分にとって価値あるものを引き出せない。だからこそ、研究室があり学生がある。しかし、そういった知の体系は、万人に開かれるべきだ</li>
<li>AIによって日々積み重なる論文や発表資料、講演録を吸収し、知の系統樹を作らせる。我々はそれが巨大な樹木となっていくのを見ながら、欠けているピースや来るべき枝葉を準備する</li>
</ul>
</li>
<li>Intel/Llama-2-70b-chat-hf-onnx-int4
<ul>
<li><a href="https://huggingface.co/Intel/Llama-2-70b-chat-hf-onnx-int4">https://huggingface.co/Intel/Llama-2-70b-chat-hf-onnx-int4</a></li>
<li>high-quality, INT4, ONNX models for all LLama2 variants (base vs. chat, 7B to 70B).</li>
</ul>
</li>
<li>Best Practices for LLM Evaluation of RAG Applications by DataBricks
<ul>
<li><a href="https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG">https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG</a></li>
<li>Human and GPT-4 judges can reach above 80% agreement on the correctness and readability score. And if we lower the requirement to be smaller or equal than 1 score difference, the agreement level can reach above 95%.</li>
</ul>
</li>
<li>Struc-Bench: Are Large Language Models Really Good at Generating Complex Structured Data?
<ul>
<li><a href="https://arxiv.org/abs/2309.08963">https://arxiv.org/abs/2309.08963</a></li>
<li>structure-aware fine-tuning method, applied to Llama-7B, which significantly outperform other model like GPT-3.5/4 and Vicuna-13B.</li>
</ul>
</li>
<li>Azure Cognitive Search のハイブリッド+セマンティックランキングは、純粋なベクターサーチよりもパフォーマンス良かったそうで！
<ul>
<li><a href="https://techcommunity.microsoft.com/t5/azure-ai-services-blog/azure-cognitive-search-outperforming-vector-search-with-hybrid/ba-p/3929167">https://techcommunity.microsoft.com/t5/azure-ai-services-blog/azure-cognitive-search-outperforming-vector-search-with-hybrid/ba-p/3929167</a></li>
</ul>
</li>
<li>“Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality”
<ul>
<li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4573321">https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4573321</a></li>
<li>① GPT-4ありの集団は以下のように優れていた ・タスクの完了数が平均で12.2%多い ・タスクの完了速度が平均で25.1%早い ・タスクの品質が平均で40%高い</li>
<li>② もともと成績のよくない人が目覚ましく向上した</li>
</ul>
</li>
<li>GPT-3.5-turbo を Fine-tuning して GPT-4 相当の性能を獲得する
<ul>
<li><a href="https://tech.drobe.co.jp/entry/2023/09/19/140000">https://tech.drobe.co.jp/entry/2023/09/19/140000</a></li>
<li>Lambda で GPT-4 を叩きつつ、入力と出力のペアを json 形式で Cloudwatch に落とします。</li>
<li>データをダウンロードしたらここを参考に Fine-tuning のデータの準備と validation を行います。</li>
<li>Fine-tuning の実施は簡単です。OpenAI の API を利用して以下を実施します。
<ul>
<li>
<ol>
<li>トレーニングデータをアップロード</li>
</ol>
</li>
<li>
<ol start="2">
<li>アップロードしたデータを指定しつつトレーニングを開始</li>
</ol>
</li>
</ul>
</li>
<li>Fine-tuning すると結果が GPT-4 に近づく事が観測できた</li>
</ul>
</li>
<li>Let’s Verify Step by Step
<ul>
<li><a href="https://arxiv.org/abs/2305.20050">https://arxiv.org/abs/2305.20050</a></li>
<li>LLMが複雑な問題を推論できるのは、学習中に推論方法（解き方）にアクセスし、その解き方を学んでいるからといえる</li>
</ul>
</li>
<li>自律言語エージェントを構築するためのフレームワーク Agents を試す by npakaさん
<ul>
<li><a href="https://note.com/npaka/n/n089614881df8">https://note.com/npaka/n/n089614881df8</a></li>
<li>「<strong>Agents</strong>」は、<strong>自律言語エージェントを構築するためのフレームワーク</strong></li>
<li>「<strong>SOP</strong>」(Standard Operation Process) を通じて言語エージェントにきめ細かい制御とガイダンスを提供できることです。「SOP」は<strong>タスク全体のサブゴール / サブタスクを定義</strong>し、ユーザーが言語エージェントのきめ細かいワークフローをカスタマイズできるようにします。</li>
</ul>
</li>
<li>Benchmarking <code>gpt-3.5-turbo-instruct</code> on agents doing question-answering over tabular data
<ul>
<li><a href="https://github.com/langchain-ai/langchain-benchmarks/blob/main/csv-qa/pandas_agent_instruct.py">https://github.com/langchain-ai/langchain-benchmarks/blob/main/csv-qa/pandas_agent_instruct.py</a></li>
<li>It performed roughly the same as gpt-3.5-turbo (the chat model) with roughly ~67% accuracy</li>
<li>It errored twice due to misformatted output - without function prompting for output format becomes much more important</li>
</ul>
</li>
<li>StableDiffusionで生成した画像から3Dモデルを"AIで"作成し、Unity上でキャラクターを動かすまで【CSM AIの使い方】
<ul>
<li><a href="https://note.com/okp_/n/n89b96384e0cb?sub_rt=share_b">https://note.com/okp_/n/n89b96384e0cb?sub_rt=share_b</a></li>
</ul>
</li>
<li>llamaindexのチュートリアル、“building RAG from scratch” -
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/low_level/root.html">https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/low_level/root.html</a></li>
</ul>
</li>
<li>SambaNova、最大5兆個のパラメータモデルを実行可能なLLM向けAI半導体「SN40L」を発表
<ul>
<li><a href="https://news.mynavi.jp/techplus/article/20230920-2775419/">https://news.mynavi.jp/techplus/article/20230920-2775419/</a></li>
<li>Ceruleanアーキテクチャ。NVIDIA H100の24台分の性能で、GPUに搭載されてる様な高速メモリが不要でメモリ大容量化が可能！DDRが使える</li>
</ul>
</li>
<li>sam altman氏、DALE 3のデモ画像を自慢する
<ul>
<li><a href="https://x.com/sama/status/1704561613070893428?s=20">https://x.com/sama/status/1704561613070893428?s=20</a></li>
</ul>
</li>
<li>OpenAI本家で、Fine-tuning用のweb pageが公開された
<ul>
<li><a href="https://x.com/OfficialLoganK/status/1704181284036300970?s=20">https://x.com/OfficialLoganK/status/1704181284036300970?s=20</a></li>
<li>誰でも簡単にモデルの微調整ができ</li>
</ul>
</li>
<li>JSONの可視化ツール jsoncrack
<ul>
<li><a href="https://jsoncrack.com/">https://jsoncrack.com/</a></li>
</ul>
</li>
<li>GPT-4などの大規模言語モデルで化学研究を行うにあたっての､現状・課題・展望を整理した論文
<ul>
<li>Prompt engineering of GPT-4 for chemical research: what can/cannot be done?</li>
<li><a href="https://www.tandfonline.com/doi/full/10.1080/27660400.2023.2260300">https://www.tandfonline.com/doi/full/10.1080/27660400.2023.2260300</a></li>
<li>GPT-4は、化学研究における言語処理やドメイン知識の組み込みに有効なツールとなり得ます。</li>
<li>以下が必要
<ul>
<li>分子構造や実験データを扱えるようにするためのプラグイン</li>
<li>マルチモーダルモデルの開発最新の化学情報を学習できるようにするためのローカルモデル</li>
<li>推論や計画能力を向上させるためのアルゴリズムやハードウェアの革新</li>
</ul>
</li>
</ul>
</li>
<li>llamaindexにて、RAGにおいて、カスタムプロンプトをつかったQueryを使う方法、<br>
- RAGStringQueryEngineというので、任意のpromptを投入できる？！<br>
- なるほどこれは役に立つ<br>
- <a href="https://gpt-index.readthedocs.io/en/latest/examples/query_engine/custom_query_engine.html">https://gpt-index.readthedocs.io/en/latest/examples/query_engine/custom_query_engine.html</a></li>
<li>An in-browser version of ChatGPT (or HF Chat), built with HuggingFace Transformers.js!<br>
- <a href="https://huggingface.co/spaces/mithril-security/blind_chat">https://huggingface.co/spaces/mithril-security/blind_chat</a><br>
- webllmとは違ったブラウザベースのlocal LLM実装、transformer.jsかあ、そっちからHF使うんだ。</li>
<li>RSJ2023「基盤モデルの実ロボット応用」チュートリアル2（松尾研）
<ul>
<li><a href="https://speakerdeck.com/tmats/rsj2023-ji-pan-moderunoshi-robotutoying-yong-tiyutoriaru2-shi-robotutoyong-noji-pan-moderuwozuo-tutehuo-yong-surufang-fa">https://speakerdeck.com/tmats/rsj2023-ji-pan-moderunoshi-robotutoying-yong-tiyutoriaru2-shi-robotutoyong-noji-pan-moderuwozuo-tutehuo-yong-surufang-fa</a></li>
<li>日本ロボット学会 <a href="https://twitter.com/hashtag/RSJ2023?src=hashtag_click">#RSJ2023</a> の「基盤モデルの実ロボット応用」セッションのチュートリアル（後半）の資料</li>
<li>基盤モデルの特徴を整理したあと，ロボティクス領域での基盤モデルを構築し活用する方法に関してサーベイ</li>
</ul>
</li>
<li><strong>Building RAG with LLMs and Prompts</strong>　by <strong>Jerry Liu, LlamaIndex</strong>
<ul>
<li>@FlowGPTOfficial workshop today I gave talks on how to build RAG response generation and a simple router module using only LLMs and prompt</li>
</ul>
</li>
<li>llamaindexのRAGにおける、類似検索語のpost processing様々、順番変えるとかありなのか・
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/node_postprocessors/modules.html#longcontextreorder">https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/node_postprocessors/modules.html#longcontextreorder</a></li>
</ul>
</li>
<li>LLMが持つ/持たない/持ちうる創造性についての論文
<ul>
<li>On the Creativity of Large Language Models</li>
<li><a href="https://arxiv.org/abs/2304.00008">https://arxiv.org/abs/2304.00008</a></li>
<li>ボーデンの３つの基準（価値、新規性、驚き）や他の哲学的理論に基づいて、LLMの創造性を検証</li>
<li>LLMは価値を持つ作品やアイデアを生成することができますが、新規性や驚きについては弱い</li>
<li>LLMは人間と同じような創造性を持っているとは言えません</li>
<li>異なる学習方法や適応能力を持つモデルを開発することで、探索的や変革的な創造性を実現することができるかもしれません</li>
<li>LLMは人間と協働することで、人間の創造性を補完したり刺激したりすることができます</li>
</ul>
</li>
<li>RAG is more than just embedding search
<ul>
<li><a href="https://jxnl.github.io/instructor/blog/2023/09/17/rag-is-more-than-just-embedding-search/">https://jxnl.github.io/instructor/blog/2023/09/17/rag-is-more-than-just-embedding-search/</a></li>
<li>シンプルなベクトルサーチベースの課題を述べながら、instructorというopenai function callingにpydanticを組み合わせられるライブラリを紹介している記事</li>
<li>課題の一つ、-   <strong>Query-Document Mismatch</strong>:ドキュメントと質問のembbedingって同じ空間でないと意味ないよね（地産地消の場合を除く）</li>
</ul>
</li>
<li>Xwin-LM-70BがGPT-4超え？
<ul>
<li><a href="https://www.itmedia.co.jp/news/articles/2309/21/news085.html">https://www.itmedia.co.jp/news/articles/2309/21/news085.html</a></li>
<li>Xwin-LMは米Metaが公開したAI「Llama2」をベースにしており、教師ありファインチューニング、報酬モデル、リジェクトサンプリング、強化学習などを使って調整したものという。パラメータ数はLlama2と同じく70億、130億、700億の3つのモデルを用意。中でも最大である700億の「Xwin-LM-70B-V0.1」は、AlpacaEvalの評価基準である「Text-Davinci-003」（GPT-3のモデルの一つ）に対する勝率で95.57％を記録。勝率95.28％のGPT-4を追い抜いたとしている。</li>
</ul>
</li>
<li>ChatGPTの知識が、2022年1月までの知識も反映した模様
<ul>
<li><a href="https://old.reddit.com/r/ChatGPT/comments/16m6yc7/gpt4_training_cutoff_date_is_now_january_2022/">https://old.reddit.com/r/ChatGPT/comments/16m6yc7/gpt4_training_cutoff_date_is_now_january_2022/</a></li>
</ul>
</li>
<li>e2e(end-to-end) LLM/RAG、RAG評価を含めてLLMでやるという話、について
<ul>
<li>raysummit2023でのチュートリアル、jupyternotebookあるよ</li>
<li><a href="https://github.com/anyscale/ray-summit-2023-training/blob/main/Ray-LlamaIndex/notebooks/02_evaluation.ipynb">https://github.com/anyscale/ray-summit-2023-training/blob/main/Ray-LlamaIndex/notebooks/02_evaluation.ipynb</a></li>
</ul>
</li>
<li>RAGを構成するときに、メタデータを与えるってのは役に立つわけだが、それをPydantic ＋LLMで一発でできるという話、
<ul>
<li>extract a full Pydantic object from any doc with 1 LLM call.</li>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/metadata_extraction/PydanticExtractor.html">https://gpt-index.readthedocs.io/en/latest/examples/metadata_extraction/PydanticExtractor.html</a></li>
</ul>
</li>
<li>Text generation web UI で Xwin-LM-13B をロードして色々推論して遊んでみます。
<ul>
<li><a href="https://note.com/sa1p/n/n51170c4d1a1f">https://note.com/sa1p/n/n51170c4d1a1f</a></li>
<li>「Text generation web UI」は、oobabooga氏による<strong>大規模言語モデル用の無料のWeb UI</strong></li>
<li>ただし、ローカルに、GPUなどが必要<strong>Windowsの場合NVIDIA製のグラボでのみ動作する</strong></li>
</ul>
</li>
<li>Exploring ReAct Agent for Better Prompting in RAG Pipeline
<ul>
<li><a href="https://betterprogramming.pub/exploring-react-agent-for-better-prompting-in-rag-pipeline-b231aae0ca7c">https://betterprogramming.pub/exploring-react-agent-for-better-prompting-in-rag-pipeline-b231aae0ca7c</a></li>
<li>use ReAct Agent to analyze Amazon’s recent disclosures and attitudes towards LLMs in their SEC Exhibits 99.1 filings</li>
</ul>
</li>
<li>RAGの評価、正解と答えとの比較評価で、従来のBLEU/ROUGEとかでなくて、単に類似性評価でよいという簡易は方法を提示
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/evaluation/semantic_similarity_eval.html">https://gpt-index.readthedocs.io/en/latest/examples/evaluation/semantic_similarity_eval.html</a></li>
</ul>
</li>
<li>OpenAI謹製の、RAG(Q&amp;A)のチュートリアル
<ul>
<li><a href="https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb">https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb</a></li>
</ul>
</li>
<li>BQMLの時系列分析、ARiMAを適当なラグ設定のもとで40モデルほどぺぺっと推定してくれて、かつAICも推定してくれるので鬼便利
<ul>
<li><a href="https://x.com/behemuhemulove/status/1705629318439907451?s=20">https://x.com/behemuhemulove/status/1705629318439907451?s=20</a></li>
</ul>
</li>
<li>LLMって何に使われているかのサーベイ
<ul>
<li><a href="https://x.com/dmvaldman/status/1705350469177295273?s=20">https://x.com/dmvaldman/status/1705350469177295273?s=20</a></li>
<li>１位：プログラムのエラーと解消法について、２位：AIのソフトウエアについての質問、３位：旅行関係、４位：テキスト要約とか改善、５位：ビジネス戦略立案</li>
</ul>
</li>
<li>GPT-3.5-Turbo-Instruct
<ul>
<li><a href="https://chatgpt-lab.com/n/n2ed70597dfbf">https://chatgpt-lab.com/n/n2ed70597dfbf</a></li>
<li>既存の「GPT-3.5-Turbo」とは違ってチャットに特化したモデルではないため、モデルが広範な自然言語処理タスクを扱うことを可能にします<br>
-OpenAIのテストでは、175Bのパラメータを持つGPTモデルよりも、1.3Bのパラメータを持つInstructGPTモデルの方が、100倍小さいにもかかわらず、人々に好まれることが示されている</li>
</ul>
</li>
</ul>
<h2 id="section-1">9/19</h2>
<p>GPT-4を活用して、データセットをつくって、他のＬＬＭをファインチューニングするとか、色々出ているが、MetaやAppleがGPT-4越えのLLMを来年に向け開発中。Appleが出遅れているのは、自動運転とかそっちにリソースを割かれているかとも、でもM2もっているし、ポテンシャルはある。あほなSiriの代わりになるのか？。RestGPTは、ReActの発展形、「APIの理解」ってのができるらしい。やっぱり企業利用ならば、RAG(Retrieval Augmented Generation)関係で、元となるテキストのチャンキングの仕方とか、ベクトルＤＢの選び方とか、スクラッチからのRAGの作成とか、地道活動も拾ってます。AstroLLaMA、今後様々なタスクや分野に特化したLLMがどんどんできてくるかも。LiteLLMっていうLLMの抽象化を使うと、アプリコードが再利用できるのか、作った人天才。GPT4による生産性向上にういての定量評価、資料として色々使えるな。仏教対話AIって、聖人をどれだけ復活させても幸せになれない気がする。きっと故人のChatBot作成サービスって葬儀業界ですぐにでも出てきそうだ。いや、2021年にマイクロソフトが<a href="https://edition.cnn.com/2021/01/27/tech/microsoft-chat-bot-patent/index.html">特許化していた</a>。。</p>
<ul>
<li>Meta、GPT-4と同程度の性能を目指すモデルの学習を計画
<ul>
<li><a href="https://www.theverge.com/2023/9/10/23867323/meta-new-ai-model-gpt-4-openai-chatbot-google-apple">https://www.theverge.com/2023/9/10/23867323/meta-new-ai-model-gpt-4-openai-chatbot-google-apple</a></li>
<li>AIトレーニングチップを買い集め、データセンターを構築</li>
<li>2024年の早い時期に新しい大規模言語モデルの学習を開始する予定</li>
<li>企業がAIツールを作成するために、再びこのモデルを無料にするよう働きかけている</li>
</ul>
</li>
<li>Fine-tuning to Memorize Knowledge
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/finetuning/knowledge/finetune_knowledge.html">https://gpt-index.readthedocs.io/en/latest/examples/finetuning/knowledge/finetune_knowledge.html</a></li>
<li>GPT4で、内部ドキュメントに対する、Q&amp;Aを生成させてこれをつかって、LLMをファイチューニングする話。</li>
<li>“bake in knowledge”と呼ぶらしい。</li>
</ul>
</li>
<li>OpenIntepreterを使っていると、OpenAIのAPIコールで10ドルが一瞬で溶ける。（デジタル庁楠さん）
<ul>
<li><a href="https://x.com/masanork/status/1701381113506329083?s=20">https://x.com/masanork/status/1701381113506329083?s=20</a></li>
<li>つまり、ChatGPT Plusの月額課金が気前いいことになっている。</li>
</ul>
</li>
<li>RestGPT: Connecting Large Language Models with Real-World RESTful APIs
<ul>
<li><a href="https://restgpt.github.io/">https://restgpt.github.io/</a></li>
<li>ReActの発展形か、、</li>
<li><a href="https://zenn.dev/carnot/articles/7f87b613a0a637">https://zenn.dev/carnot/articles/7f87b613a0a637</a>
<ul>
<li><strong>言語のみの指示から複数のAPIを呼び出すことが可能</strong></li>
<li>RestGPTではプランニング・APIの理解・APIの選択をそれぞれのモジュールが独立で行うため、複雑なユーザ要求にも柔軟に対応することが可能になっています。</li>
</ul>
</li>
</ul>
</li>
<li>来年にはGPT-4を上回る能力を持つとされる３つのモデル
<ul>
<li>① OpenAI: GPT-4.5/GPT5</li>
<li>② Google: Gemini</li>
<li>③ Apple: Ajax</li>
<li>Apple is reportedly spending ‘millions of dollars a day’ training AI</li>
<li><a href="https://www.theverge.com/2023/9/6/23861763/apple-ai-language-models-ajax-gpt-training-spending">https://www.theverge.com/2023/9/6/23861763/apple-ai-language-models-ajax-gpt-training-spending</a></li>
</ul>
</li>
<li>仏教対話AIの多様化に成功―親鸞ボットと菩薩ボットの増産―(京大）
<ul>
<li><a href="https://www.kyoto-u.ac.jp/ja/research-news/2023-09-12-0">https://www.kyoto-u.ac.jp/ja/research-news/2023-09-12-0</a></li>
<li>生成系AI「ChatGPT 4」と宗教を掛け合わせた新型チャットボット「親鸞ボット」と「世親ボット」を共同開発し、仏教対話AIの多様化に成功しました。</li>
<li><a href="https://www.itmedia.co.jp/news/articles/2309/14/news083.html">会話事例</a>が、地獄にしか見えないのは気のせい？</li>
</ul>
</li>
<li>リクルートにおける数理最適化の 活用事例と産学連携の取り組み
<ul>
<li><a href="https://speakerdeck.com/recruitengineers/rikurutoniokerushu-li-zui-shi-hua-no-huo-yong-shi-li-tochan-xue-lian-xi-noqu-rizu-mi">https://speakerdeck.com/recruitengineers/rikurutoniokerushu-li-zui-shi-hua-no-huo-yong-shi-li-tochan-xue-lian-xi-noqu-rizu-mi</a></li>
<li>企業における数理最適化専門グループって、大変なのよね。</li>
</ul>
</li>
<li>生成AIブームで多発の可能性　「PoC貧乏」
<ul>
<li><a href="https://forbesjapan.com/articles/detail/65744/page2">https://forbesjapan.com/articles/detail/65744/page2</a></li>
<li>「生成AIで何かビジネスを作ってみて」と上層部が丸投げし、成果が出ないまま人件費がかさむ、ゆるやかなPoC貧乏が頻発することが考えられます。</li>
<li>まあ、生成AIに限らないわけだが。。</li>
</ul>
</li>
<li>Calls out of chaos: the adaptive significance of nonlinear phenomena in mammalian vocal production
<ul>
<li><a href="https://www.sciencedirect.com/science/article/abs/pii/S0003347201919128">https://www.sciencedirect.com/science/article/abs/pii/S0003347201919128</a></li>
<li>赤子の泣き声がカオス的なダイナミクスで、複雑さと予測不可能性によって親に無視させないようにする適応的意義があるらしい</li>
</ul>
</li>
<li>自然言語処理で扱うテキストのchunkingについて
<ul>
<li><a href="https://zenn.dev/hijikix/articles/f414b067e29a57">https://zenn.dev/hijikix/articles/f414b067e29a57</a></li>
<li>Adjacent Sequence Clustering</li>
<li>全体の文章をセンテンスに分割した後、チャンクに詰めていくのだが、その際に直前のセンテンスと処理中のセンテンスの意味的類似度を比較して、意味が離れているものは次のチャンクに詰める</li>
</ul>
</li>
<li>llamaindexのRAG作成チュートリアル（ローレベル）
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/low_level/root.html">https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/low_level/root.html</a></li>
<li>ローレベルというのは、プリミティブな処理で構成するという意味。</li>
</ul>
</li>
<li>llamaindexのResponseの作り方
<ul>
<li>Building Response Synthesis from Scratch</li>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/low_level/response_synthesis.html">https://gpt-index.readthedocs.io/en/latest/examples/low_level/response_synthesis.html</a></li>
<li>promptをカスタマイズできるのが素敵。</li>
</ul>
</li>
<li>Vector databases (Part 4): Analyzing the trade-offs
<ul>
<li><a href="https://thedataquarry.com/posts/vector-db-4/">https://thedataquarry.com/posts/vector-db-4/</a></li>
<li>ベクトルDBのトレードオフを分析した記事。挿入vs読取速度、取りこぼし（Recall）vsレイテンシー、インメモリvsオンディスク、全文検索vsベクトルハイブリッド検索等の観点から比較・分析を実質</li>
</ul>
</li>
<li>AstroLLaMA: Towards Specialized Foundation Models in Astronomy
<ul>
<li><a href="https://arxiv.org/abs/2309.06126">https://arxiv.org/abs/2309.06126</a></li>
<li>特定分野に特化したLLMが大量発生する予感。</li>
</ul>
</li>
<li>東京都の 「文章生成AI利活用ガイドライン」
<ul>
<li><a href="https://www.metro.tokyo.lg.jp/tosei/hodohappyo/press/2023/08/23/14.html">https://www.metro.tokyo.lg.jp/tosei/hodohappyo/press/2023/08/23/14.html</a></li>
<li>プロンプトの具体例も豊富でわかりやすい</li>
</ul>
</li>
<li>llamaindexがLiteLLMをサポート、＋１００のLLｍが利用可能に？？
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/stable/examples/llm/litellm.html">https://gpt-index.readthedocs.io/en/stable/examples/llm/litellm.html</a></li>
<li>(OpenAI, Cohere, AnthropicAI, huggingface, etc.)に対して同じインターフェイスを提供。</li>
<li>というか、LiteLLMすごいな。</li>
</ul>
</li>
<li>Announcing the Preview of OpenAI Whisper in Azure OpenAI service and Azure AI Speech
<ul>
<li><a href="https://techcommunity.microsoft.com/t5/azure-ai-services-blog/announcing-the-preview-of-openai-whisper-in-azure-openai-service/ba-p/3928388">https://techcommunity.microsoft.com/t5/azure-ai-services-blog/announcing-the-preview-of-openai-whisper-in-azure-openai-service/ba-p/3928388</a></li>
<li>Azure OpenAIサービスおよびAzure AI SpeechでのOpenAI Whisperのプレビューを発表しました</li>
</ul>
</li>
<li>Discover the LLMs
<ul>
<li><a href="https://llm.extractum.io/">https://llm.extractum.io/</a></li>
<li>LLM の VRAM や Context Len が一覧表示できて便利</li>
</ul>
</li>
<li>BCGとハーバードやMIT等によるGPT4を使用したタスク実験
<ul>
<li>Centaurs and Cyborgs on the Jagged Frontier</li>
<li><a href="https://www.oneusefulthing.org/p/centaurs-and-cyborgs-on-the-jagged">https://www.oneusefulthing.org/p/centaurs-and-cyborgs-on-the-jagged</a></li>
<li>BCGのコンサルティング758名で実験</li>
<li>18種類のコンサルタスクが対象</li>
<li>AIを使用したコンサルは 、12.2％多く仕事を終え、 25.1％早く仕事を完了し、 40％高い品質</li>
</ul>
</li>
<li>Optimizing LLMs From a Dataset Perspective
<ul>
<li><a href="https://sebastianraschka.com/blog/2023/optimizing-LLMs-dataset-perspective.html">https://sebastianraschka.com/blog/2023/optimizing-LLMs-dataset-perspective.html</a></li>
<li>LLMsの最適化について、データセットの側面からまとめたブログ。人手で高品質なデータセットを作るグループや、LLMから大量のデータセットを生成するグループなど、いくつかの側面が簡潔にまとまっている</li>
</ul>
</li>
<li>InstaGraph
<ul>
<li><a href="https://github.com/yoheinakajima/instagraph">https://github.com/yoheinakajima/instagraph</a></li>
<li>任意のドキュメントから知識グラフ作れるらしい。</li>
<li>例：<a href="https://x.com/yoheinakajima/status/1701351068817301922?s=20">https://x.com/yoheinakajima/status/1701351068817301922?s=20</a></li>
</ul>
</li>
</ul>
<h2 id="section-2">9/11</h2>
<p>8/23に公開されたGPT-3.5-turboのfine-tuning API、RAGとの比較、証券報告書のQ&amp;Aアプリの具体例、など、面白い記事がたくさん出てきた。Open Interpreterも相も変わらず熱い。デジタル庁のChatGPTの業務利用ハンズオン、いいな、こういうリテラシーを持てる人が増えないと。。大規模コンテンツ・行動モデル（LCBM）って、記号接地問題にさらに近づこうとしているのか？LLMをつかった様々なエージェントの作り方、いろんなデータ専門のエージェントがたくさんそろってくると、そろそろOrchestratorが必要かな。<strong>Production-Ready LLM Applications</strong>ってのは必読なスライドですね。ICML2023のまとめもあった。RAGを対象としたLLMの比較、フレームワークになってありがたい。ChatGPTの複数出力とか、性能が落ちたのでは？という疑惑など、何が起きているのか、起こそうとしているのか。</p>
<ul>
<li>東京大学理学部オープンキャンパス2023 講演「生成型AIの数理と倫理」佐藤一誠教授
<ul>
<li><a href="https://www.youtube.com/watch?v=n6NDlgJVug8&amp;t=5s">https://www.youtube.com/watch?v=n6NDlgJVug8&amp;t=5s</a></li>
</ul>
</li>
<li>Mustafa Suleyman on getting Washington and Silicon Valley to tame AI
<ul>
<li><a href="https://80000hours.org/podcast/episodes/mustafa-suleyman-getting-washington-and-silicon-valley-to-tame-ai/">https://80000hours.org/podcast/episodes/mustafa-suleyman-getting-washington-and-silicon-valley-to-tame-ai/</a></li>
<li>DeepMindの共同創業者で、世界最高水準のAIスパコンを構築中のAI開発会社「Inflection AI」の設立者でもあるスレイマン氏によれば、今後18ヶ月程度でGPT-4の学習に使用された計算回数の10倍〜100倍がAIモデルの学習に使用され、次の3年程度でGPT-4の1000倍の計算回数が学習に使われるだろう、とのこと</li>
</ul>
</li>
<li>LangChain Cheat Sheet
<ul>
<li><a href="https://www.kdnuggets.com/2023/08/langchain-cheat-sheet.html">https://www.kdnuggets.com/2023/08/langchain-cheat-sheet.html</a></li>
</ul>
</li>
<li>llamaindexより、Summary Index(旧List Index)の紹介
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/stable/core_modules/data_modules/index/index_guide.html#summary-index-formerly-list-index">https://gpt-index.readthedocs.io/en/stable/core_modules/data_modules/index/index_guide.html#summary-index-formerly-list-index</a></li>
</ul>
</li>
<li>AI Agents – Build and Host LLM Apps At Scale
<ul>
<li>LLMを活用さいた様々なエージェントの作り方についての記事、なるほど</li>
<li><a href="https://blog.abacus.ai/blog/2023/08/31/supercharge-productivity-accomplish-10x-more-with-ai-agents/">https://blog.abacus.ai/blog/2023/08/31/supercharge-productivity-accomplish-10x-more-with-ai-agents/</a></li>
</ul>
</li>
<li>Large Content And Behavior Models To Understand, Simulate, And Optimize Content And Behavior
<ul>
<li><a href="https://huggingface.co/papers/2309.00359">https://huggingface.co/papers/2309.00359</a></li>
<li><strong>大規模コンテンツ・行動モデル（LCBM）とコンテンツ・行動コーパス（CBC）</strong>：本文書では、行動トークンをLLMの訓練に再導入する初期的な試みを行う。LCBMと呼ばれる新しいモデルは、コンテンツ理解タスクにおいてLLMと同等の性能を示すとともに、行動シミュレーション、コンテンツシミュレーション、行動理解、行動ドメイン適応といった能力も持つ。さらに、LCBMの研究を促進するために、コミュニケーター、メッセージ、受信者行動を含む新しいコーパスであるCBCを公開する。</li>
</ul>
</li>
<li>ChatGPTを業務に組み込むためのハンズオン
<ul>
<li>デジタル庁が一般公開しているChatGPTの入門</li>
<li><a href="https://www.digital.go.jp/assets/contents/node/information/field_ref_resources/5896883b-cc5a-4c5a-b610-eb32b0f4c175/82ccd074/20230725_resources_ai_outline.pdf">https://www.digital.go.jp/assets/contents/node/information/field_ref_resources/5896883b-cc5a-4c5a-b610-eb32b0f4c175/82ccd074/20230725_resources_ai_outline.pdf</a></li>
<li>なかなかのやり手が書いている、ここまで試行できる人は少ないのでは？？</li>
<li>プロンプトの書き方のコツ
<ul>
<li>できる限りコンテキストを明確にして書くこと</li>
<li>GPTの理解度(?)を確認しながら進める</li>
<li>最初はマニュアルを読むより、まず自分でやってみて感覚をつかみことを推奨</li>
</ul>
</li>
</ul>
</li>
<li>最近のLLMの学習法のまとめ - SFT・RLHF・RAG　by npakaさん、
<ul>
<li><a href="https://note.com/npaka/n/n862786604dc3">https://note.com/npaka/n/n862786604dc3</a></li>
<li>とりあえず、どれだけ知ってる？だけでもリトマス試験紙になる、むろん私はRAG派</li>
<li>SFT : Supervised Fine-Tuning</li>
<li>RLHF : Reinforcement Learning from Human Feedback</li>
<li>RAG : Retrieval Augmented Generation</li>
</ul>
</li>
<li>LangChain を使ったRAGを Elyza 7b instruct モデル
<ul>
<li><a href="https://note.com/alexweberk/n/n3cffc010e9e9">https://note.com/alexweberk/n/n3cffc010e9e9</a></li>
<li>無料のT4ではメモリーオーバーで動かないんだが。。。</li>
</ul>
</li>
<li>SEC Insights
<ul>
<li>llamaindexを活用して、米国証券取引委員会への報告書(SEC-10)にたいするQ&amp;Aアプリを作る例</li>
<li><a href="https://github.com/run-llama/sec-insights">https://github.com/run-llama/sec-insights</a></li>
<li><a href="https://www.secinsights.ai/">https://www.secinsights.ai/</a></li>
</ul>
</li>
<li>Streamlit 入門  by npakaさん
<ul>
<li><a href="https://note.com/npaka/n/n29b5e8088fe5">https://note.com/npaka/n/n29b5e8088fe5</a></li>
<li>「Streamlit」は、機械学習およびデータサイエンスのためのWebアプリケーションフレームを簡単に作成して共有できるPythonライブラリ</li>
<li>もうちょっとどうにかならんのか？</li>
</ul>
</li>
<li><strong>Production-Ready LLM Applications</strong>
<ul>
<li>llamaindexのCEOより、</li>
<li><a href="https://docs.google.com/presentation/d/1uzhz1aFWbyXSrWBzQ1FPQWtVjMgJqAYGoGoVzEnNmAg/edit#slide=id.p">https://docs.google.com/presentation/d/1uzhz1aFWbyXSrWBzQ1FPQWtVjMgJqAYGoGoVzEnNmAg/edit#slide=id.p</a>
<ul>
<li>Fine-tuning: LLMs + embeddings</li>
<li>Better Data + Retrieval Techniques for Production RAG</li>
</ul>
</li>
</ul>
</li>
<li>ELYZA-7bは、M1 MacBook Airでもサクサク動くらしい
<ul>
<li><a href="https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf/blob/main/README.md">https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf/blob/main/README.md</a></li>
</ul>
</li>
<li>やっぱりOpenInterpreterが熱い
<ul>
<li><a href="https://github.com/KillianLucas/open-interpreter">https://github.com/KillianLucas/open-interpreter</a></li>
</ul>
</li>
<li>LLMをホストするAnyScaleのllamaindexでの利用例
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/llm/anyscale.html">https://gpt-index.readthedocs.io/en/latest/examples/llm/anyscale.html</a></li>
<li>run + finetune open-source LLMs through an API</li>
<li>そういうビジネスができるのか。。</li>
</ul>
</li>
<li>Fine-Tuning GPT-3.5 RAG Pipeline with GPT-4 Training Data
<ul>
<li><a href="https://betterprogramming.pub/fine-tuning-gpt-3-5-rag-pipeline-with-gpt-4-training-data-49ac0c099919">https://betterprogramming.pub/fine-tuning-gpt-3-5-rag-pipeline-with-gpt-4-training-data-49ac0c099919</a></li>
<li>どうも、8/23にOpenAIがGPT-3.5-turboのfine-tuning APIを公開して、即座にllmaindexがこれに対応したらしい</li>
<li>じゃあ、Q&amp;Aアプリを作るのに、RAGとFine-tuningどちらが高性能か？ということへの考察記事</li>
<li>こちらは、llamaindexをつかったGPT-3.5-turboのfine-tuningのcolab
<ul>
<li><a href="https://colab.research.google.com/drive/1NgyCJVyrC2xcZ5lxt2frTU862v6eJHlc?usp=sharing">https://colab.research.google.com/drive/1NgyCJVyrC2xcZ5lxt2frTU862v6eJHlc?usp=sharing</a></li>
</ul>
</li>
</ul>
</li>
<li>Hierachical Agent
<ul>
<li>対象ドキュメントの内容が階層構造であるような場合のQ&amp;Aの作り方。</li>
<li><a href="https://colab.research.google.com/drive/1qIb09SyuLeiwGy_FGcRcQpM78yQ2p0_3?usp=sharing">https://colab.research.google.com/drive/1qIb09SyuLeiwGy_FGcRcQpM78yQ2p0_3?usp=sharing</a></li>
</ul>
</li>
<li>Discover LlamaIndex: Custom Tools for Data Agent
<ul>
<li><a href="https://www.youtube.com/watch?v=lcuL6Gqw_-g">https://www.youtube.com/watch?v=lcuL6Gqw_-g</a></li>
</ul>
</li>
<li>【速報】OpenAI APIでGPT-3.5-turboがfine-tuningできるようになりました！
<ul>
<li><a href="https://dev.classmethod.jp/articles/openai-gpt35turbo-fine-tuning/">https://dev.classmethod.jp/articles/openai-gpt35turbo-fine-tuning/</a></li>
<li>学習するサンプルは最小10個必要で、50～100個で明確な改善が見られる</li>
<li>gpt-3.5-turboでfine-tuningが利用可能に</li>
<li>gpt-3のモデルであるbabbage-002とdavinci-002も新しいfine-tuningでサポート（モデルもGPT baseという扱い）</li>
</ul>
</li>
<li>グラフニューラルネットの 2023年まとめ (ICML2023)
<ul>
<li>軽量 Transformer の介入や Diffusion for Molecules などの実世界利用、幾何学的な利用が記載されている</li>
<li><a href="https://towardsdatascience.com/graph-machine-learning-icml-2023-9b5e4306a1cc">https://towardsdatascience.com/graph-machine-learning-icml-2023-9b5e4306a1cc</a></li>
</ul>
</li>
<li>Open Inerpreterの利用例、「nikkei225の10年分をプロットして」と滅入れすればあとは自動で、、、
<ul>
<li><a href="https://twitter.com/NuCode/status/1700679106814501132?s=20">https://twitter.com/NuCode/status/1700679106814501132?s=20</a></li>
</ul>
</li>
<li>ChatGPTが、可能性のある答えを複数ていじするようになった、RLHFやらせようとしているのかと話題に
<ul>
<li><a href="https://twitter.com/GrantSlatton/status/1700662574315090351?s=20">https://twitter.com/GrantSlatton/status/1700662574315090351?s=20</a></li>
</ul>
</li>
<li>LLMの評価、特にRetrieval Augmented Generation (RAG) パイプラインを評価するためのOSSフレームワークragas
<ul>
<li><a href="https://github.com/explodinggradients/ragas">https://github.com/explodinggradients/ragas</a></li>
</ul>
</li>
<li>Agent deconstructedに、llmaindex agentが統合された？
<ul>
<li><a href="https://github.com/shoggoth13/agents-deconstructed/blob/main/notebooks/react_chat.ipynb">https://github.com/shoggoth13/agents-deconstructed/blob/main/notebooks/react_chat.ipynb</a></li>
<li>ReActができるようになったのか。。、いろんなindexをもつLLM同士が会話して問題解決。。</li>
</ul>
</li>
<li>【デモ付き】Embeddingsで独自データをChatGPTに理解させる
<ul>
<li><a href="https://corp.langcore.org/media/embeddings">https://corp.langcore.org/media/embeddings</a></li>
<li>LangCore SaaSを使ってインフラ不要で手軽にEmbeddingsを活用した独自データの活用、らしい</li>
</ul>
</li>
</ul>
<h2 id="section-3">9/4</h2>
<p>GoogeからGPT-4対抗のGeminiが発表、GPT-4 の 2023 倍の計算能力を持つ？。LLMのファインチューニング関係で、様々な紹介がある。llamaindex周りの記事が多いが、それだけRAG(Retrieval-Augmented Generation)って需要があるということか。Embeddingもしっかり性能評価やファインチューニングすると性能があたる。llamaindexでQ&amp;Aの性能を上げるためのTipsが詳しく書いてある、これは役立つ。ローカルLLMの試行も熱い、なんとCode interpreterもどきも動くという。最近のLLMでは、ELYZAが一番の模様(by shi3z)。理論関係では、transformerにおける自己注意はSVMと等価なのか？、確率過程の新刊も気になる。</p>
<ul>
<li>LLMのファインチューニング で 何ができて 何ができないのか
<ul>
<li><a href="https://note.com/npaka/n/nec63c01f7ee8">https://note.com/npaka/n/nec63c01f7ee8</a></li>
</ul>
</li>
<li>code llama がhuggingfaceのchatに登場
<ul>
<li><a href="https://huggingface.co/chat/">https://huggingface.co/chat/</a></li>
</ul>
</li>
<li>Llamaで、出力を指定するためのgrammar-based sampling
<ul>
<li><a href="https://python.langchain.com/docs/integrations/llms/llamacpp#grammars">https://python.langchain.com/docs/integrations/llms/llamacpp#grammars</a></li>
</ul>
</li>
<li>Google 「Gemini」は、ChatGPT-4 Enterprise プラットフォームの直接の競合相手
<ul>
<li><a href="https://www.theinformation.com/articles/the-forced-marriage-at-the-heart-of-googles-ai-race">https://www.theinformation.com/articles/the-forced-marriage-at-the-heart-of-googles-ai-race</a></li>
<li>GPT-4 の 2023 倍の計算能力を持つ</li>
</ul>
</li>
<li>llamainexでembeddingをファインチューニングする
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/finetuning/embeddings/finetune_embedding.html">https://gpt-index.readthedocs.io/en/latest/examples/finetuning/embeddings/finetune_embedding.html</a></li>
</ul>
</li>
<li>論文紹介 / Llama 2: Open Foundation and Fine-Tuned Chat Models　by NTT西田さん
<ul>
<li><a href="https://speakerdeck.com/kyoun/llama-2-open-foundation-and-fine-tuned-chat-models">https://speakerdeck.com/kyoun/llama-2-open-foundation-and-fine-tuned-chat-models</a></li>
</ul>
</li>
<li>ご家庭用LLMでストリーミングする方法
<ul>
<li><a href="https://note.com/shi3zblog/n/n66ae41af7c64">https://note.com/shi3zblog/n/n66ae41af7c64</a></li>
<li>“elyza/ELYZA-japanese-Llama-2-7b-instruct”　利用</li>
</ul>
</li>
<li>LlamaIndexの性能向上のためのテクニックガイド by npaka
<ul>
<li><a href="https://note.com/npaka/n/n33e28a9e1409">https://note.com/npaka/n/n33e28a9e1409</a></li>
</ul>
</li>
<li>Discover LlamaIndex: Introduction to Data Agents for Developers
<ul>
<li><a href="https://www.youtube.com/watch?v=GkIEEdIErm8">https://www.youtube.com/watch?v=GkIEEdIErm8</a></li>
<li>first-ever video tutorial on LlamaIndex Data Agents</li>
</ul>
</li>
<li>ChatGPT vs BERT：どちらが日本語をより理解できるのか？
<ul>
<li><a href="https://fintan.jp/page/9126/">https://fintan.jp/page/9126/</a></li>
</ul>
</li>
<li>LlamaIndex の QAプロンプト と Refineプロンプト のカスタマイズ
<ul>
<li><a href="https://note.com/npaka/n/ne878095d5bda">https://note.com/npaka/n/ne878095d5bda</a></li>
</ul>
</li>
<li>llama2-13b-128k、論文を全部理解して要約を吐き出す方法
<ul>
<li><a href="https://gist.github.com/alfredplpl/33fd6dd6d623d4da959f1ca8aabc88fe">https://gist.github.com/alfredplpl/33fd6dd6d623d4da959f1ca8aabc88fe</a></li>
</ul>
</li>
<li>「データ分析のための統計学入門」
<ul>
<li><a href="http://www.kunitomo-lab.sakura.ne.jp/2021-3-3Open(S).pdf">http://www.kunitomo-lab.sakura.ne.jp/2021-3-3Open(S).pdf</a></li>
</ul>
</li>
<li>【ローカルLLM】text-generation-webUIのAPI機能を試す
<ul>
<li><a href="https://note.com/bakushu/n/na4e51d377ae7">https://note.com/bakushu/n/na4e51d377ae7</a></li>
<li>LLM用のウェブUIであるtext-generation-webUIにAPI機能が付属しているので、これを使ってExllama＋GPTQのAPIを試してみた。</li>
</ul>
</li>
<li>最近のLLMの性格 by shi3z
<ul>
<li><a href="https://twitter.com/madyagi/status/1697949115190255951?s=20">https://twitter.com/madyagi/status/1697949115190255951?s=20</a></li>
<li>ELYZAが良いみたい。</li>
</ul>
</li>
<li>Transformers as Support Vector Machines
<ul>
<li><a href="https://arxiv.org/abs/2308.16898">https://arxiv.org/abs/2308.16898</a></li>
</ul>
</li>
<li>fine-tuned a gpt-3.5 ReAct agent to be better at chain-of-thought
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/finetuning/react_agent/react_agent_finetune.html">https://gpt-index.readthedocs.io/en/latest/examples/finetuning/react_agent/react_agent_finetune.html</a></li>
</ul>
</li>
<li>機械学習のための確率過程入門
<ul>
<li><a href="https://www.ohmsha.co.jp/book/9784274231087/">https://www.ohmsha.co.jp/book/9784274231087/</a></li>
</ul>
</li>
<li>ローカルPCのターミナル上でLLM生成コードを実行できるOpen Interpreterを試す
<ul>
<li><a href="https://note.com/hamachi_jp/n/n05ae28b76d9d">https://note.com/hamachi_jp/n/n05ae28b76d9d</a></li>
<li>ChatGPTのコードインタープリター（Advanced Data Analysis）と同様な機能をローカル環境で実行可能な Open Interpreter</li>
<li>llamaに差し替えることも可能</li>
</ul>
</li>
<li></li>
</ul>
<h2 id="section-4">8/28</h2>
<p>先週発表された、松尾研の“Weblab-10B”に対する量子化やローカル環境での実行も花開くが、やっぱり今週はメタによるCode Llamaの発表がポイントになっている。<br>
「LLM によるプログラムベース推論」的な考え方ってLLMをつかったアプリ作成には絶対必須な考え方になると思う。品質保証では、ガードレールとか、推論過程のガイドが必要だったり、得手不得手をちゃんと理解したうえでガイドするみたいな感じ。emergent機能とはLLMを動かしていて、予測していたのとは違う機能が創発するという話、欧州ＡＩ規制でも言及される、仕組みの解明と対策が急務。llamaindexから、外部検索と組み合わせる新しい、Metaphor機能がリリース。なんかどこのURLを見ればよいかのDBをつかってやるみたいな感じ。。HuggingFaceでは、LLMをWebベースで、ファインチューニングできる機能が公開されたらしい。結果はそのままHuggingFaceに乗るみたいなノリ。LLMをつかったQ&amp;AであるRAGフレームワークで、類似データをtop-kでとってくる仕組みがうまくいかないときの工夫など、納得感ある。メタからCode Llamaが発表、コード生成ができる。さっそく、量子化されたり、llama.cppでローカルに動かしたりと、あっというまに、誰でも使えるようになる。コミュニティはすごいな。理論面では、emergentスキルに関して、通常の汎化理論に反する「スリングショット汎化」の提唱、ＬＬＭをつかった帰納的学習法というのも、従来の予測を書き換えるか。ＡＩ規制に対するパブコメをＡＩで分析など面白いかも。。</p>
<ul>
<li>言語モデルにおける複雑なスキルの創発に関する理論　A Theory for Emergence of Complex Skills in Language Models
<ul>
<li><a href="https://note.com/daichi_mu/n/n72b6265b09f6">https://note.com/daichi_mu/n/n72b6265b09f6</a></li>
<li>言語モデルのスケールアップに伴う新たなスキルの出現について、統計的枠組みと数学的分析を用いて分析する。能力レベルが通常の汎化理論に反する「スリングショット汎化」の概念を導入</li>
</ul>
</li>
<li>LLM によるプログラムベース推論
<ul>
<li><a href="https://speakerdeck.com/smiyawaki0820/2023-dot-08-dot-07-geography-and-language-mian-qiang-hui-number-4">https://speakerdeck.com/smiyawaki0820/2023-dot-08-dot-07-geography-and-language-mian-qiang-hui-number-4</a></li>
<li>LLM 開発における評価・品質担保に関係、ガードレールや、推論過程のガイドなど最後はVisProg紹介</li>
<li>東北大の宮脇さん、地理空間情報をLLMをつかいながら推論する仕組みについて。</li>
</ul>
</li>
<li>AIが「理解」するから、API仕様書のコピペでアプリができあがるローコード開発環境「Flowise」
<ul>
<li><a href="https://internet.watch.impress.co.jp/docs/column/shimizu/1523766.html">https://internet.watch.impress.co.jp/docs/column/shimizu/1523766.html</a></li>
</ul>
</li>
<li><strong><a href="https://github.com/sotokisehiro/chatux-server-llm">chatux-server-llm</a></strong>
<ul>
<li>ローカル環境で動作する文章生成 AI チャットボットです。 CPU だけで動作します。</li>
<li>LINE の japanese-large-lm-3.6b-instruction-sft を CTranslate2 化</li>
</ul>
</li>
<li>Vicuna 13B v1.5 、text-generation-webui じゃなくて以前試作した llama.cpp の HTTP サーバー機能を使ってみたら普通に LLaMA 2 13B と遜色ない結果出してくれた
<ul>
<li><a href="https://twitter.com/izutorishima/status/1693468524222861589?s=20">https://twitter.com/izutorishima/status/1693468524222861589?s=20</a></li>
</ul>
</li>
<li>Metaの大規模言語モデル「LLaMA」のトレーニングにも使用されたAIの学習用データセット「Books3」が削除される
<ul>
<li><a href="https://gigazine.net/news/20230821-books-3-ai-data-set/">https://gigazine.net/news/20230821-books-3-ai-data-set/</a></li>
<li>知的財産権や著作権に対する侵害の疑いが指摘されていたらしい</li>
</ul>
</li>
<li>LlamaIndex + Metaphor: Towards Automating Knowledge Work with LLMs
<ul>
<li><a href="https://medium.com/llamaindex-blog/llamaindex-metaphor-towards-automating-knowledge-work-with-llms-5520a32efa2f">https://medium.com/llamaindex-blog/llamaindex-metaphor-towards-automating-knowledge-work-with-llms-5520a32efa2f</a></li>
<li>Metaphor was trained to predict links on the internet, given how people talk about things on the Internet</li>
<li>インターネット検索とllamaindexの融合？結合の新たな形としてのメタファー？</li>
</ul>
</li>
<li>【ローカルLLM】Gradio+CTranslate2で日本語LLMのチャットUIをつくる
<ul>
<li><a href="https://note.com/bakushu/n/nba6e9c353ee4">https://note.com/bakushu/n/nba6e9c353ee4</a></li>
<li><a href="https://huggingface.co/line-corporation/japanese-large-lm-3.6b-instruction-sft">line-corp-japanese-large-lm-3.6b</a>を利用</li>
<li>CTranslate2で量子化</li>
<li>あとはgradioでWebUI生成！</li>
</ul>
</li>
<li>Generally Intelligence社、米国商務省国家電気通信情報庁（NTIA）が実施したAI規制に関するパブリックコメントの約1450件の回答の分析を開始。
<ul>
<li><a href="https://generallyintelligent.com/perspectives/ntia-rfc-analysis/">https://generallyintelligent.com/perspectives/ntia-rfc-analysis/</a></li>
<li><a href="https://twitter.com/kanjun/status/1693819078866354376?s=20">https://twitter.com/kanjun/status/1693819078866354376?s=20</a></li>
</ul>
</li>
<li>llamaindexのMetaphorサーチのお試しができるらしい。
<ul>
<li><a href="https://twitter.com/jerryjliu0/status/1693773766797746649?s=20">https://twitter.com/jerryjliu0/status/1693773766797746649?s=20</a></li>
<li><a href="https://colab.research.google.com/drive/1PTnJTVmLAI-V8JJu8GsbUvbk8vs203kA?usp=sharing">https://colab.research.google.com/drive/1PTnJTVmLAI-V8JJu8GsbUvbk8vs203kA?usp=sharing</a></li>
</ul>
</li>
<li>Stanford大学のHAIから、Create AI Actを連邦政府が法案をとおすべきである、米国のため
<ul>
<li>We Must Pass the Create AI Act</li>
<li><a href="https://hai.stanford.edu/news/we-must-pass-create-ai-act?utm_source=twitter&amp;utm_medium=social&amp;utm_content=Stanford%20HAI_twitter_StanfordHAI_202308220803_sf181078680&amp;utm_campaign=&amp;sf181078680=1">https://hai.stanford.edu/news/we-must-pass-create-ai-act?utm_source=twitter&amp;utm_medium=social&amp;utm_content=Stanford HAI_twitter_StanfordHAI_202308220803_sf181078680&amp;utm_campaign=&amp;sf181078680=1</a></li>
</ul>
</li>
<li>Open AI でスーパーアライメントを4年以内に完了させることを目標として率いているJan Leike氏の対談
<ul>
<li><a href="https://80000hours.org/podcast/episodes/jan-leike-superalignment/">https://80000hours.org/podcast/episodes/jan-leike-superalignment/</a></li>
</ul>
</li>
<li>Inductive-bias Learning: Generating Code Models with Large Language Model
<ul>
<li><strong>帰納的学習法</strong>：大規模言語モデル（LLM）を用いて、説明変数から目的変数を予測するモデルを生成する新しい学習法。この学習法は、教師あり学習とメタラーニングの要素を持つ。</li>
<li><a href="https://arxiv.org/abs/2308.09890">https://arxiv.org/abs/2308.09890</a></li>
</ul>
</li>
<li>日本語が使えるようになったGoogle PaLM2を試す
<ul>
<li><a href="https://note.com/eurekachan/n/n62b15394b5dc">https://note.com/eurekachan/n/n62b15394b5dc</a></li>
<li>BigQuery のSQLなんかも日本語で生成をお願いすることが出来ます。</li>
<li>LangChainからも呼び出したりできるようです</li>
</ul>
</li>
<li>ANYONE can fine-tune (almost) any LLM available on Hugging Face
<ul>
<li>Hugging Faceで簡単にLLMをファインチューニングできるAPIが公開</li>
<li><a href="https://twitter.com/abhi1thakur/status/1693619860050153958?s=20">https://twitter.com/abhi1thakur/status/1693619860050153958?s=20</a></li>
</ul>
</li>
<li>RAGシステムで、top-k 抽出がうまくいかないときの工夫について
<ul>
<li><a href="https://twitter.com/jerryjliu0/status/1694013501323563101?s=20">https://twitter.com/jerryjliu0/status/1694013501323563101?s=20</a></li>
<li>Metadata Filters + Auto Retrieval:</li>
<li>Store Document Hierarchies (summaries -&gt; raw chunks) + Recursive Retrieval</li>
</ul>
</li>
<li>今村・松井の『ベイズ最適化』
<ul>
<li>第4章までよめらば、ベイズ最適化が理解できるらしい。</li>
<li><a href="https://www.kindaikagaku.co.jp/book_list/detail/9784764906631/">https://www.kindaikagaku.co.jp/book_list/detail/9784764906631/</a></li>
</ul>
</li>
<li>メタが、Code Llamaを公表
<ul>
<li><a href="https://ai.meta.com/blog/code-llama-large-language-model-coding/">https://ai.meta.com/blog/code-llama-large-language-model-coding/</a></li>
<li>Foundation base models (Code Llama)</li>
<li>Python specializations (Code Llama - Python),</li>
<li>Instruction-following models (Code Llama - Instruct)</li>
</ul>
</li>
<li>【ローカルLLM】Colabの標準GPUで「CodeLlama-34B-GGUF」を動かす
<ul>
<li><a href="https://note.com/bakushu/n/n21cb30a15f27">https://note.com/bakushu/n/n21cb30a15f27</a></li>
<li>量子化は「GPTQ」ではなくて、CPU＋GPUで実行できる「GGUF(旧GGML)」</li>
<li>標準GPU（Tesla T4）で動くのがみそ</li>
</ul>
</li>
<li>Weblab-10Bを量子化(GPTQ)して簡単に動かすことがhugging faceでできる
<ul>
<li>transformersにGPTQが統合されたおかげで、無料Colabでそのままでは動かなかったWeblab-10Bもらくらく動くようになってた。</li>
<li>dahara1/weblab-10b-instruction-sft-GPTQ</li>
<li><a href="https://github.com/webbigdata-jp/python_sample/blob/main/weblab_10b_instruction_sft_GPTQ_sample.ipynb">https://github.com/webbigdata-jp/python_sample/blob/main/weblab_10b_instruction_sft_GPTQ_sample.ipynb</a></li>
</ul>
</li>
<li>【まとめ】Google Colab で Code Llama を試す
<ul>
<li><a href="https://note.com/npaka/n/n51ed424b2943">https://note.com/npaka/n/n51ed424b2943</a></li>
</ul>
</li>
<li>CodeLlama model now work w/ llama-cpp-python
<ul>
<li><a href="https://twitter.com/TheBlokeAI">@TheBlokeAI</a>さんによる</li>
<li>llama.cpp GGUFの組み合わせで動くということ</li>
<li><a href="https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/tree/main">https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/tree/main</a></li>
<li><a href="https://github.com/abetlen/llama-cpp-python">https://github.com/abetlen/llama-cpp-python</a></li>
</ul>
</li>
<li>CodeLamaの、colabでの実行とビデオ
<ul>
<li><a href="https://colab.research.google.com/drive/1lyEj1SRw0B9I2UUI2HOrtiJ_fjvbXtA2?usp=sharing">https://colab.research.google.com/drive/1lyEj1SRw0B9I2UUI2HOrtiJ_fjvbXtA2?usp=sharing</a></li>
<li><a href="https://www.youtube.com/watch?v=rlCe_lG4uhk">https://www.youtube.com/watch?v=rlCe_lG4uhk</a></li>
</ul>
</li>
</ul>
<h2 id="section-5">8/21</h2>
<p>暑くて溶けそうなのに、電力はどうにかもっている夏です。松尾研からの国産LLMである“Weblab-10B”の発表。なお、松尾研には夏休み中の総理も訪問され講座を受講（なにか修了証書をもらってたな）、もっと国としてのサポートが期待できるかも。GPT-4は、暗号化されたプロンプトも理解できるぐらい優れているらしいが、特定の「脱獄プロンプト」に弱い面も。Trustworthy LLM、LLMの信頼性などの研究も進む、社会規範への整合とかそういう側面もある。スタンフォード大学のLLMの安全性のベンチマークとの比較も気になる。あいもかわらず知識グラフ系のLLM応用がちらほら、知識グラフ抽出や知識グラフをつかったRAG(Retrieval-Augmented Generation)などもあるが、知識の活用かそれともファインチューニングか？みたいな第２世代(エキスパートシステム）と第３世代（データがすべて）のAIの対比みたいな絵面だなあ。MRIスペクトルから分子を予想みたいな素朴な応用がもっとあっていい気もする。TRL(Transformer Reinforcement Learning)は、強化学習を用いたLLMの最適化を簡単にできるようになるらしい、DPO(Direct Preference Optimization)なんか斬新じゃん。元Googleトップ研究者による「Sakana AI」にはびっくり、めざす「自然からインスピレーションを得たインテリジェンスに基づいた新しいタイプの基礎モデル」とはどんなものになるのか？日本はコンテンツだけでなくて、人材リソースとしてもまだ魅力がある？？</p>
<ul>
<li>ローカルデータに対するQ&amp;Aなどするときに、知識を活用したRAGで構成するのがよいのか、いや、目的に対してLLMをファインチューニングするがいいのかというはなし
<ul>
<li>Knowledge Graphs &amp; LLMs: Fine-Tuning vs. Retrieval-Augmented Generation</li>
<li><a href="https://neo4j.com/developer-blog/fine-tuning-retrieval-augmented-generation/">https://neo4j.com/developer-blog/fine-tuning-retrieval-augmented-generation/</a></li>
</ul>
</li>
<li>LLMをつかったsemantic searchのDeeplearning.aiの無料コース
<ul>
<li>Large Language Models with Semantic Search</li>
<li><a href="https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/">https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/</a></li>
</ul>
</li>
<li>GPT-4のセーフガードを故意に突破する脱獄プロンプトに関する研究
<ul>
<li>“Do Anything Now”: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models</li>
<li><a href="https://jailbreak-llms.xinyueshen.me/">https://jailbreak-llms.xinyueshen.me/</a></li>
</ul>
</li>
<li>「汎用的なAIってやつ」を作ったところで、それで十分なレベルまで収益化を実現させるのはそれなりに難しいという話(TJO
<ul>
<li><a href="https://twitter.com/TJO_datasci/status/1691112696685719553">https://twitter.com/TJO_datasci/status/1691112696685719553</a></li>
</ul>
</li>
<li>GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher
<ul>
<li><a href="https://arxiv.org/abs/2308.06463">https://arxiv.org/abs/2308.06463</a></li>
<li>GPT-4 can understand ciphertext, which introduces the risk of generating unsafe content.</li>
</ul>
</li>
<li>CSVにたいするQ&amp;Aエージェントのベンチマーク
<ul>
<li><a href="https://github.com/langchain-ai/langchain-benchmarks/tree/main/csv-qa">https://github.com/langchain-ai/langchain-benchmarks/tree/main/csv-qa</a></li>
</ul>
</li>
<li>Knowledge Graph RAG Query Engine (RAG: Retrieval-Augmented Generation)
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/query_engine/knowledge_graph_rag_query_engine.html">https://gpt-index.readthedocs.io/en/latest/examples/query_engine/knowledge_graph_rag_query_engine.html</a></li>
<li>augmenting LLMs with context from a graph database</li>
</ul>
</li>
<li>Large Language Models with Semantic Search
<ul>
<li><a href="https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/">https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/</a></li>
<li>Deeplearing.aiからのsemantic searchの無料コース、Cohereの人がでている？</li>
</ul>
</li>
<li>知識グラフ抽出のデモ
<ul>
<li>text to graph playground</li>
<li><a href="https://auto-graph.streamlit.app/">https://auto-graph.streamlit.app/</a></li>
</ul>
</li>
<li>Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language Models’ Alignment
<ul>
<li>LLMの信頼性に関するサーベイ論文</li>
<li>用語や概念を整理し，実際に8つの観点からLLMの信頼性を検証</li>
<li><a href="https://arxiv.org/abs/2308.05374">https://arxiv.org/abs/2308.05374</a></li>
<li>reliability, safety, fairness, resistance to misuse, explainability and reasoning, adherence to social norms, and robustness.</li>
<li>目的は：reliable and ethically sound deployment of LLMs in various applications.</li>
</ul>
</li>
<li>RWKVについて解説
<ul>
<li><a href="https://agirobots.com/rwkv/">https://agirobots.com/rwkv/</a></li>
<li>RNNの利点である高速な推論と処理可能なシーケンス長を大幅に向上</li>
</ul>
</li>
<li>LLMに関して起きている訴訟について
<ul>
<li><a href="https://twitter.com/srush_nlp/status/1691845245074620915?s=20">https://twitter.com/srush_nlp/status/1691845245074620915?s=20</a></li>
</ul>
</li>
<li>LLMでMRIスペクトルから分子を予測
<ul>
<li><a href="https://chemrxiv.org/engage/chemrxiv/article-details/64d5e4ccdfabaf06ff1763ef">https://chemrxiv.org/engage/chemrxiv/article-details/64d5e4ccdfabaf06ff1763ef</a></li>
<li>NMRスペクトルを文字列で表現、これを言語モデルへ入力し分子を予測することで67%の精度</li>
</ul>
</li>
<li>松尾研究室100億パラメータサイズ・日英2ヶ国語対応の大規模言語モデル“Weblab-10B”をオープンソースで公開
<ul>
<li><a href="https://weblab.t.u-tokyo.ac.jp/100%E5%84%84%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%BA%E3%83%BB%E6%97%A5%E8%8B%B12%E3%83%B6%E5%9B%BD%E8%AA%9E%E5%AF%BE%E5%BF%9C%E3%81%AE%E5%A4%A7%E8%A6%8F%E6%A8%A1/">https://weblab.t.u-tokyo.ac.jp/100億パラメータサイズ・日英2ヶ国語対応の大規模/</a></li>
<li><a href="https://huggingface.co/matsuo-lab/weblab-10b">https://huggingface.co/matsuo-lab/weblab-10b</a></li>
<li>日本語のベンチマークであるJGLUE評価値が事前学習時と比べて大幅に改善（66→78%</li>
<li>早速オープンソース警察が、商用に使えないのにオープンソースとは言わないとの突っ込みが。。</li>
</ul>
</li>
<li>岸田首相、東京大で生成AIの講座受ける　「百聞は一見にしかず」
<ul>
<li><a href="https://www.asahi.com/articles/ASR8G6X84R8GUTFK002.html">https://www.asahi.com/articles/ASR8G6X84R8GUTFK002.html</a></li>
<li>松尾豊・東大大学院教授の講座を受けた。AIを学習させるプログラミングも体験し、受講</li>
</ul>
</li>
<li>LLMをつかった文書検索では、メタデータを入れることで性能が改善する
<ul>
<li>Building Production-Ready LLM Apps with LlamaIndex: Document Metadata for Higher Accuracy Retrieval</li>
<li><a href="https://betterprogramming.pub/building-production-ready-llm-apps-with-llamaindex-document-metadata-for-higher-accuracy-retrieval-a8ceca641fb5">https://betterprogramming.pub/building-production-ready-llm-apps-with-llamaindex-document-metadata-for-higher-accuracy-retrieval-a8ceca641fb5</a></li>
</ul>
</li>
<li>GoogleのトップAI研究者2人が東京でAI企業立ち上げを発表
<ul>
<li>「自然からインスピレーションを得たインテリジェンスに基づいた新しいタイプの基礎モデルを開発する」</li>
<li>ジョーンズ氏とハー氏が新AI企業「Sakana AI」を東京に設立</li>
<li>うち1人は、生成AI革命のきっかけとなった論文の著者の一人</li>
<li>日本で研究者を募り、生成AIの基盤モデル開発を目指す</li>
<li><a href="https://www.nikkei.com/article/DGXZQOUC186TM0Y3A810C2000000/?n_cid=SNSTW001&amp;n_tw=1692351448">https://www.nikkei.com/article/DGXZQOUC186TM0Y3A810C2000000/?n_cid=SNSTW001&amp;n_tw=1692351448</a></li>
<li>起業の地に日本を選んだ理由として、米国で生成AIの人材獲得競争が過熱している点をあげた。</li>
</ul>
</li>
<li>TRL - 強化学習によるLLMの学習のためのライブラリ
<ul>
<li>TRL - Transformer Reinforcement Learning</li>
<li><a href="https://note.com/npaka/n/nbb974324d6e1">https://note.com/npaka/n/nbb974324d6e1</a></li>
<li>強化学習を使用してTransformer言語モデルを学習できます。このライブラリはHuggingFace Transformersと統合されています。</li>
</ul>
</li>
<li>DPO による Llama 2 のファインチューニング(npaka)
<ul>
<li><a href="https://note.com/npaka/n/nfe7391a1d28d">https://note.com/npaka/n/nfe7391a1d28d</a></li>
<li>「Direct Preference Optimization」では、既存の手法で使用されているRLベースの目標を、単純なバイナリクロスエントロピー損失を介して直接最適化できる目標に切り替える</li>
<li>LMを改良するこのプロセスが大幅に簡素化</li>
</ul>
</li>
</ul>
<h2 id="section-6">8/14</h2>
<p>お盆ですが、膨大にならないうちに更新します。ところで、「大規模言語モデル入門」(技術評論社ISBN 978-4-297-13633-8）いいですね、Huggingfacesをつかって、日本語データセットをつかった、ファインチューニングなど見所が多い。<br>
さて今週は、先週に引き続き vicuna-v1.5関係の記事が多かったわけですが、stability.aiから日本語のStableLLMがリリースされたがのが大きなニュースでした。LLMベンチマークもColab環境でできるらしい。Metaの公表した生成AIのガイドとか、FacToolなんか、AIの安全性やリスクなんかに対してちゃんと取り組んでいる。日FR本のAI戦略の、開発促進に偏った姿勢とは一線を画している（つまり余裕がないということ）。FacToolによる分析の結果、GPT-4はやっぱりすごいんだな。Llmaindexのllmがgpt-3.5-turboにやっと変更されたらしい、そんなに使いにくかったのか。。LLMをプロダクションで使うための色々なTipsが公表されてたり、一方Andrew Ngさんは、LLMが世界を理解しているというブログを開陳。LLM時代の医療へのAI利用のベネフィットとリスクについてのランサー記事とか、数学者Terence Taoさんの、LLMをつかったAIが数学論文の共著者になりうるという興味深い予測も。産総研のAIセミナー、あっという間に満杯に。興味だけは大きいのに、手が動かない人が多すぎないか。。まあ、LLMでいくら頑張てもChatGPTでよくない？みたいな意見もある。様々な面で、日本はLLM開発で遅れてきていて、もはや以前のような横綱相撲をするような感じではないのに政府はそうは言えないのか、しかし民間は頑張っている。</p>
<ul>
<li>LlamaIndexでAutoGPTQモデルを使う（vicuna-13B-v1.5-GPTQ）
<ul>
<li><a href="https://zenn.dev/libratech/articles/1979874b223895">https://zenn.dev/libratech/articles/1979874b223895</a></li>
<li>4bit化など軽量化されたllmをllamaindexで使う方法、ローカル環境とか</li>
<li>Colabの無料版(T4インスタンス)でも動作する</li>
</ul>
</li>
<li>LLama2公開にあわせて、Metaから"responsible generative AI"に関するガイドが出ている.、
<ul>
<li><a href="https://ai.meta.com/static-resource/responsible-use-guide/">https://ai.meta.com/static-resource/responsible-use-guide/</a></li>
</ul>
</li>
<li>text-generation-webui で TheBloke/vicuna-13B-v1.5-GPTQが動く
<ul>
<li><a href="https://twitter.com/smorce1/status/1688250856129646592?s=20">https://twitter.com/smorce1/status/1688250856129646592?s=20</a></li>
</ul>
</li>
<li>llama2をつかって、ローカルにQ&amp;Aを実行する手法について via llamaindex
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/vector_stores/SimpleIndexDemoLlama-Local.html">https://gpt-index.readthedocs.io/en/latest/examples/vector_stores/SimpleIndexDemoLlama-Local.html</a></li>
</ul>
</li>
<li>LLMを試すのに、「ガンダムテスト」というのがあるらしい、vicuna-13b-v1.5-16kは優秀らしい
<ul>
<li><a href="https://twitter.com/NuCode/status/1688455649091608576?s=20">https://twitter.com/NuCode/status/1688455649091608576?s=20</a></li>
</ul>
</li>
<li>内閣府AI戦略会議(8/4)の資料が一部公開、AI関連施策は開発振興一本足に近くリスク対応が申し訳程度
<ul>
<li><a href="https://www8.cao.go.jp/cstp/ai/ai_senryaku/4kai/shisaku.pdf">https://www8.cao.go.jp/cstp/ai/ai_senryaku/4kai/shisaku.pdf</a></li>
</ul>
</li>
<li>IPA「ITパスポート試験 シラバス」に、生成AIの仕組み、活用例、留意事項等に関する項目・用語例を追加
<ul>
<li><a href="https://www.ipa.go.jp/shiken/syllabus/henkou/2023/20230807.html">https://www.ipa.go.jp/shiken/syllabus/henkou/2023/20230807.html</a></li>
</ul>
</li>
<li>「JP Language Model Evaluation Harness」によるLLM性能評価 by stabilityAI
<ul>
<li><a href="https://note.com/npaka/n/nedf4dacd4037">https://note.com/npaka/n/nedf4dacd4037</a></li>
<li>Colab(T4)で12時間もかかる、できるらしい</li>
</ul>
</li>
<li>llama-2-13bのJGLUE、言語モデルの評価と関係
<ul>
<li><a href="https://huggingface.co/HachiML/Llama-2-13b-hf-qlora-dolly-ja-2ep/blob/main/benchmark_jglue/JGLUE_Llama-2-13b-hf-qlora-dolly-ja-2ep.ipynb">https://huggingface.co/HachiML/Llama-2-13b-hf-qlora-dolly-ja-2ep/blob/main/benchmark_jglue/JGLUE_Llama-2-13b-hf-qlora-dolly-ja-2ep.ipynb</a></li>
</ul>
</li>
<li>GPTQの元論文はこちら、
<ul>
<li><a href="https://arxiv.org/pdf/2210.17323.pdf">https://arxiv.org/pdf/2210.17323.pdf</a></li>
<li>GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS</li>
</ul>
</li>
<li>ストックマークは最近の話題にも詳しいGPT-NeoXをベースとした14億パラメータの日本語のLLMをOSS公開
<ul>
<li><a href="https://stockmark.co.jp/news/20230808">https://stockmark.co.jp/news/20230808</a></li>
</ul>
</li>
<li>HuggingFacesとNVIDIAが提携、企業向けのサービスを展開？
<ul>
<li><a href="https://www.nvidia.com/ja-jp/about-nvidia/press-releases/2023/nvidia-and-hugging-face-to-connect-millions-of-developers-to-generative-ai-supercomputing/">https://www.nvidia.com/ja-jp/about-nvidia/press-releases/2023/nvidia-and-hugging-face-to-connect-millions-of-developers-to-generative-ai-supercomputing/</a></li>
<li>HuggingFaceにあるAIモデルのトレーニングとか微調整ができる企業向けのサービスで、GPUとしてNVidiaのクラウドGPUが選べるようになるらしい。</li>
</ul>
</li>
<li>悲報？：産総研、LLMのセミナー「シミュレーションとAIの融合技術とその最新事例」、すぐに定員いっぱいになる
<ul>
<li><a href="https://www.airc.aist.go.jp/seminar_detail/seminar_069.html">https://www.airc.aist.go.jp/seminar_detail/seminar_069.html</a></li>
</ul>
</li>
<li><a href="http://Stability.ai">Stability.ai</a>、 日本語言語モデル「Japanese StableLM Alpha」をリリース(8/10)
<ul>
<li><a href="https://ja.stability.ai/blog/japanese-stablelm-alpha">https://ja.stability.ai/blog/japanese-stablelm-alpha</a></li>
</ul>
</li>
<li>早速Japanese Stable LLMを、Colab無料環境から利用するnotebookが公開
<ul>
<li><a href="https://colab.research.google.com/github/mkshing/notebooks/blob/main/stabilityai_japanese_stablelm_alpha_7b.ipynb">https://colab.research.google.com/github/mkshing/notebooks/blob/main/stabilityai_japanese_stablelm_alpha_7b.ipynb</a></li>
<li>huggingfacesにログインしないといけない、、が動くぞ！</li>
<li>ガンダムテストしてみたが、なんか、学習時につかったデータが表示される。</li>
</ul>
</li>
<li>生成AIによって生成されたテキストを判別する方法についての論文
<ul>
<li><a href="https://arxiv.org/abs/2306.15666">https://arxiv.org/abs/2306.15666</a></li>
<li>Testing of Detection Tools for AI-Generated Text</li>
<li>■文章のスタイルを変化させられている場合（例えば子供っぽくなど）、識別が困難になる</li>
<li>■言い換えや書き換えによって段階的に文章を変更されると、識別がかなり困難になる</li>
<li>■AI生成コードの検出はAI生成テキストの検出よりもさらに困難になる</li>
</ul>
</li>
<li>Langchainのテキスト分割の様子を目視できる、playgroundが爆誕
<ul>
<li><a href="https://langchain-text-splitter.streamlit.app/">https://langchain-text-splitter.streamlit.app/</a></li>
</ul>
</li>
<li>Google Colab で Japanese StableLM Alpha + LlamaIndex の QA を試す
<ul>
<li><a href="https://note.com/npaka/n/n5c80ca661357">https://note.com/npaka/n/n5c80ca661357</a></li>
</ul>
</li>
<li>「とっきょ」広報誌で、こち亀の内容が、拒絶通知の理由になった事例が紹介。。
<ul>
<li><a href="https://www.jpo.go.jp/news/koho/kohoshi/vol57/07_page1.html">https://www.jpo.go.jp/news/koho/kohoshi/vol57/07_page1.html</a></li>
<li>拒絶を避けるべく、特許出願する前にはこち亀を全巻読破する必要があるのか、、、</li>
<li>審査官の趣味という気もするが、、</li>
</ul>
</li>
<li>ChatGPTの新機能カスタム指示の面白い使い方
<ul>
<li><a href="https://note.com/it_navi/n/nca4643390969">https://note.com/it_navi/n/nca4643390969</a></li>
<li>カスタム指示は、ChatGPTの<strong>役割、回答方針、出力形式など</strong>を予め設定することができます。</li>
</ul>
</li>
<li>LLMは世界を理解しているか？by Andrew Ng
<ul>
<li><a href="https://www.deeplearning.ai/the-batch/issue-209/">https://www.deeplearning.ai/the-batch/issue-209/</a></li>
<li>Othelo-GPTの例から、答えは YESらしい。</li>
</ul>
</li>
<li>生成AIの文章やコード、論文が“事実か”チェックする技術　米Meta含む研究者らが開発
<ul>
<li><a href="https://www.itmedia.co.jp/news/articles/2308/09/news064.html">https://www.itmedia.co.jp/news/articles/2308/09/news064.html</a></li>
<li>FacTool: Factuality Detection in Generative AI – A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios</li>
<li><a href="https://arxiv.org/abs/2307.13528v2">https://arxiv.org/abs/2307.13528v2</a>
<ul>
<li>研究者らはベンチマークを開発し、知識ベースのQA、コード生成、数学の問題解決、科学論文のレビュー執筆の4つのタスクで実験を行った。その結果、GPT-4はChatGPT、Bard、Claude-v1、Vicunaと比較して、事実精度が最も優れていた。Vicuna-13Bは、知識ベースのQAではそれなりに良好な事実性を示したが、コード生成、数学の問題解決、科学論文のレビュー執筆など、より困難なシナリオではパフォーマンスが低い結果となった。</li>
</ul>
</li>
</ul>
</li>
<li>llamaindexのv0.8がリリース
<ul>
<li><a href="https://github.com/jerryjliu/llama_index/blob/main/CHANGELOG.md">https://github.com/jerryjliu/llama_index/blob/main/CHANGELOG.md</a></li>
<li>[1] The default LLM is now gpt-3.5-turbo</li>
<li>[2] Speaking of changing prompts, we’ve changed the default question-answering templates for both our create and refine strategy as well as tree_summarize.</li>
<li>[3] Our default text splitter is now our brand-new sentence text splitter.</li>
<li>[4] Added llama.cpp and @huggingface as fallbacks if openai key is not set.</li>
<li>[5] Some new features: a <code>SentenceWindowNodeParser</code> and <code>MetadataReplacementNodPostProcessor</code></li>
</ul>
</li>
<li>チュートリアル、Create a CustomGPT And Supercharge your Company with AI – Pick the Best LLM
<ul>
<li><a href="https://blog.abacus.ai/blog/2023/08/10/create-your-custom-chatgpt-pick-the-best-llm-that-works-for-you/">https://blog.abacus.ai/blog/2023/08/10/create-your-custom-chatgpt-pick-the-best-llm-that-works-for-you/</a></li>
</ul>
</li>
<li>Building LLM applications for production
<ul>
<li><a href="https://huyenchip.com/2023/04/11/llm-engineering.html">https://huyenchip.com/2023/04/11/llm-engineering.html</a></li>
<li>LLMをプロダクションで使うための色々なTipsがまとまった記事</li>
</ul>
</li>
<li>いろいろLLMをいじってみても、結局ChatGPTでよくない？みたいな
<ul>
<li><a href="https://twitter.com/mr_bay_area/status/1689868431900975104?s=20">https://twitter.com/mr_bay_area/status/1689868431900975104?s=20</a></li>
</ul>
</li>
<li>AI in medicine: creating a safe and equitable future
<ul>
<li>Lancerの記事、LLM時代における、医療分野へのAI適用のメリットとリスクについてまとめ</li>
<li><a href="https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(23)01668-9/fulltext">https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(23)01668-9/fulltext</a></li>
</ul>
</li>
<li>Embracing change and resetting expectations by Terence Tao@microsoft
<ul>
<li><a href="https://unlocked.microsoft.com/ai-anthology/terence-tao/">https://unlocked.microsoft.com/ai-anthology/terence-tao/</a></li>
<li>He predicts that AI will be a trustworthy co-author in mathematical research by 2026, when combined with search and symbolic math tools.</li>
<li>2026年までには、数学研究において、AIが信頼できる共著者になりうるとの予測</li>
</ul>
</li>
</ul>
<h2 id="section-7">8/7</h2>
<p>llama2ベースのVicuna v1.5で盛り上がっている、langchainやllamaindexとの組み合わせでも動く模様。ReActなどのAgent機能もちゃんとうごくらしい。llama2を手ごろに試せるcolab noteもたくさん公開、ローカルGPUで動かす報告も。なおllama2本家も申請すればを直接使うこともできる。マイクロソフトはwindows上でのllama2というネタでメタとパートナーとのこと、二股かけてる？マイクロソフトがAzure OpenAIをつかったChatGPTもどきのサンプル実装を公開、カニばってない？文章から知識を抽出する方法、llamaindexでも知識グラフ(KG)を抽出するKnowledgeGraphIndexがあったが、REBELという外部のtransformerを利用する方法もあるのか。用途に合わせて選択、細かい調整が必要かな。UCバークレーのDynalang、AIエージェントと紹介されているが、論文タイトルからするとLLMで世界モデルを構築しようとしている（「二重過程モデル」の真ん中に出てくるやつ？記号接地モデルというかそういうやつ）。自コンテンツをつかったChatBotの作り方についてわかりやすい説明があった。JSTの生成AIのまとめ、日本の生きる道は、「第4世代AI」「信頼されるAI」「AI・データ駆動科学」ということらしい。「第４世代AI」とはSystem1とSystem2が連動する、「二重過程モデル」のことらしい、Dynalangの話ともつながった！</p>
<blockquote>
<p>NeurIPS2019で、Bengioの基調講演の「二重過程モデル」（即時的なSystem1と熟考的なSystem2の二重モデル、間に、世界モデルが入る）。知覚系の深層学習(System1)によって眼前の状況に対する世界モデル（World Model）が得られるが、それを使って言語・知識系が適切な手順を組み立てるのがSystem2。カーネマンのFast &amp; Slowとも関連がありそう。。</p>
</blockquote>
<ul>
<li>Google Colab で Vicuna-v1.5 + LlamaIndex の QA を試す
<ul>
<li>npakaさんより、ハイメモリでないと動かないのか。。</li>
<li><a href="https://note.com/npaka/n/n931319f17b34">https://note.com/npaka/n/n931319f17b34</a></li>
</ul>
</li>
<li>Google Colab で Llama 2 + LlamaIndex の QA を試す
<ul>
<li>npakaさんより、llma2利用には申請が必要なのか、</li>
<li>Q&amp;Aテンプレに修正が必要なもよう</li>
<li><a href="https://note.com/npaka/n/n3e1b59d1ac9e">https://note.com/npaka/n/n3e1b59d1ac9e</a></li>
</ul>
</li>
<li>vicuna-7b-v1.5の一番簡単な利用方法by npakaさｎ
<ul>
<li><a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">https://huggingface.co/lmsys/vicuna-7b-v1.5</a></li>
<li><a href="https://twitter.com/npaka123/status/1686872443305295878?s=20">https://twitter.com/npaka123/status/1686872443305295878?s=20</a></li>
</ul>
</li>
<li>ChatGPTの小改良が順次リリースされるとの告知
<ul>
<li><a href="https://twitter.com/OpenAI/status/1687159114047291392?s=20">https://twitter.com/OpenAI/status/1687159114047291392?s=20</a></li>
<li>prompt exampleとか、Plus会員にはGPT-4がデフォルトになるとか、そういうｙつ</li>
</ul>
</li>
<li>UCバークレー、アルファ碁とChatGPTを混ぜて強くしたようなAIエージェント「Dynalang」
<ul>
<li><a href="https://arxiv.org/abs/2308.01399">https://arxiv.org/abs/2308.01399</a></li>
<li>Learning to Model the World with Language</li>
</ul>
</li>
<li>マイクロソフト社、Azure OpenAIで、ChatGPTもどきを作るサンプル実装を公開
<ul>
<li><a href="https://github.com/microsoft/azurechatgpt">https://github.com/microsoft/azurechatgpt</a></li>
<li>企業利用が加速するか。。いやplaygroundで十分？</li>
</ul>
</li>
<li>人工知能研究の新潮流2　～基盤モデル・生成AIのインパクト～
<ul>
<li>JSTのまとめ、生成AI研究の動向報告書</li>
<li><a href="https://www.jst.go.jp/crds/report/CRDS-FY2023-RR-02.html?fbclid=IwAR0KQ7bg5BRLIblzI154AHYheNrF1SPPzm-xn4z1PuQBUPK2Kia2qT4PMxU">https://www.jst.go.jp/crds/report/CRDS-FY2023-RR-02.html?fbclid=IwAR0KQ7bg5BRLIblzI154AHYheNrF1SPPzm-xn4z1PuQBUPK2Kia2qT4PMxU</a></li>
<li>「第4世代AI」「信頼されるAI」「AI・データ駆動科学」</li>
</ul>
</li>
<li>雇用判断にAIを使うのは、EU規制上禁止？
<ul>
<li>禁止ではなくて、ハイリスクAIに相当するから、守るべきことを守らないといけないということ</li>
<li><a href="https://twitter.com/umiyuki_ai/status/1687639267273748480?s=20">https://twitter.com/umiyuki_ai/status/1687639267273748480?s=20</a></li>
</ul>
</li>
<li>南極の氷が、今年は急激にとけているらしい　via 安宅さん
<ul>
<li><a href="https://www.economist.com/graphic-detail/2023/08/02/the-rapid-loss-of-antarctic-sea-ice-brings-grim-scenarios-into-view">https://www.economist.com/graphic-detail/2023/08/02/the-rapid-loss-of-antarctic-sea-ice-brings-grim-scenarios-into-view</a></li>
</ul>
</li>
<li>REBELという関係抽出トランスフォーマーをつかって知識グラフを抽出して推論する例
<ul>
<li><a href="https://twitter.com/jerryjliu0/status/1687607838539927553?s=20">https://twitter.com/jerryjliu0/status/1687607838539927553?s=20</a></li>
<li>llamaindexの人による紹介、なんか抽出する知識の密度を調整したいところ</li>
</ul>
</li>
<li>Google Colab で LangChain + Vicuna-v1.5 のエージェント機能を試す
<ul>
<li><a href="https://note.com/npaka/n/nb3c02ce2d4c5">https://note.com/npaka/n/nb3c02ce2d4c5</a></li>
<li>npakaさんより、serpAIとmathをツールとして、ReActが試せるらしい。ハイメモリが必要。。</li>
</ul>
</li>
<li>Google Colab で Llama.cpp + Vicuna-v1.5 を試す
<ul>
<li>npakaさんより、Colabでこんなこともできるのか？</li>
<li><a href="https://note.com/npaka/n/n280ffc0d5ff0">https://note.com/npaka/n/n280ffc0d5ff0</a></li>
</ul>
</li>
<li>llama-2-7bをつかって、colabでchatbodを作る例、
<ul>
<li>動くんだ、、、というか動くぞ！</li>
<li><a href="https://colab.research.google.com/github/camenduru/text-generation-webui-colab/blob/main/llama-2-7b-chat.ipynb">https://colab.research.google.com/github/camenduru/text-generation-webui-colab/blob/main/llama-2-7b-chat.ipynb</a></li>
</ul>
</li>
<li>自分のコンテンツを学習したカスタムChatBotを作る方法
<ul>
<li><a href="https://zenn.dev/karaage0703/articles/c8baa66c40f9b7">https://zenn.dev/karaage0703/articles/c8baa66c40f9b7</a></li>
<li>そうか、いつもやってるやつは、Retrieval-Augmented Generation（RAG）ってよばれているのか？</li>
</ul>
</li>
<li>LLMがローカルで動くパラメータ数どこまで？Metaの「Llama 2」を試してみた
<ul>
<li><a href="https://pc.watch.impress.co.jp/docs/column/nishikawa/1519390.html">https://pc.watch.impress.co.jp/docs/column/nishikawa/1519390.html</a></li>
<li>西川さんが組むとは、だいぶ民主化が進んだのか。</li>
<li>Colabでも結構簡単にうごくが、ローカルなGeForce RTX 4070 Ti(12GB)でも動かす事例が(西川 和久)</li>
</ul>
</li>
<li>Llama 2ベースのLLM FastChat/Vicuna v1.5をローカルで動作
<ul>
<li><a href="https://jweb.asia/26-it/ai/91-fastchat-vicuna-v1-5-on-llama-2.html">https://jweb.asia/26-it/ai/91-fastchat-vicuna-v1-5-on-llama-2.html</a></li>
</ul>
</li>
</ul>
<h2 id="section-8">7/31</h2>
<p>いやあ、暑くなって１週間さぼったら、それなりにまとめるのがつらい。メタのLLaMa2リリースが大きな話題、岡野原さんの解説が良いかも。さっそくggml化、webui対応、LanChain組み込みが行われる。LangChainの統合開発環境LangSmith、よくLangChainの紹介動画に出てきてやつが正式リリースか。メタはマイクロソフトと組んでOSS化するとのこと、マイクロソフト無敵だな。OpanAI x Azureの人は、マイクロソフトの「ChatGPT - Azure OpenAI 大全」は参考になるか。ChatGPTの性能が初期に比べて劣化しているとの報告も。「生成AIと著作権に関する論点整理」の図は素晴らしい。OpenAIのCEOであるSam Altman氏が共同創業したWorldCoinプロジェクトが7/24に仮想通貨WLDをローンチした、日本にも虹彩認証Orbが複数設置されるも認知度は今一歩か、AIで得られた利益を配る、BIプロジェクトの一旦とのこと。</p>
<ul>
<li>LLaMa2をリリース、商用利用が可能に
<ul>
<li><a href="https://ai.meta.com/llama/">https://ai.meta.com/llama/</a></li>
</ul>
</li>
<li>LLaMa2を早速ggmlに変換された
<ul>
<li><a href="https://huggingface.co/TheBloke">https://huggingface.co/TheBloke</a></li>
</ul>
</li>
<li>メタ社LLaMa2を、Microsoftと組んでOSS化すると発表
<ul>
<li><a href="https://twitter.com/alex_valaitis/status/1681348531834044426?s=20">https://twitter.com/alex_valaitis/status/1681348531834044426?s=20</a></li>
</ul>
</li>
<li>Llama2-70B-Chatモデルは、なんと有用性評価でGPT-3.5TurboのChatGPTを打倒！
<ul>
<li><a href="https://twitter.com/umiyuki_ai/status/1681361453838929923?s=20">https://twitter.com/umiyuki_ai/status/1681361453838929923?s=20</a></li>
</ul>
</li>
<li>LangChaiの統合開発環境LangSmith正式版発表
<ul>
<li><a href="https://blog.langchain.dev/announcing-langsmith/">https://blog.langchain.dev/announcing-langsmith/</a></li>
<li>おっと、正式発表されたのか</li>
</ul>
</li>
<li>Llama2は学習データを2Tトークンに増やしコンテキスト長を4KにしGQAを採用。報告書では有用性と安全性の向上に向けたSFTとRLHFの詳細が充実している。
<ul>
<li>岡野原さんの解説</li>
<li><a href="https://twitter.com/hillbig/status/1681436336451125257?s=20">https://twitter.com/hillbig/status/1681436336451125257?s=20</a></li>
</ul>
</li>
<li>BigChat Enterpriseを発表
<ul>
<li><a href="https://blogs.microsoft.com/blog/2023/07/18/furthering-our-ai-ambitions-announcing-bing-chat-enterprise-and-microsoft-365-copilot-pricing/">https://blogs.microsoft.com/blog/2023/07/18/furthering-our-ai-ambitions-announcing-bing-chat-enterprise-and-microsoft-365-copilot-pricing/</a></li>
<li>ユーザーとビジネスデータは暗号化され、組織外に流れることはありません。またチャット履歴は保存されずMicrosoftから見れません</li>
</ul>
</li>
<li>LLaMA2、ネット上のデモだとあんま日本語強くない印象だけど、ローカルでggml 4bit版の13B chat動かした感じ想像以上にまともに会話できるな、という印象
<ul>
<li><a href="https://twitter.com/RosaRugosaBeach/status/1681554704701194240?s=20">https://twitter.com/RosaRugosaBeach/status/1681554704701194240?s=20</a></li>
</ul>
</li>
<li>東大の大規模言語モデルサマースクール
<ul>
<li><a href="https://deeplearning.jp/llm2023/">https://deeplearning.jp/llm2023/</a></li>
</ul>
</li>
<li>ChatGPTの性能が、初期リリースに比べて最近低下しているとの論文が
<ul>
<li><a href="https://arxiv.org/pdf/2307.09009.pdf">https://arxiv.org/pdf/2307.09009.pdf</a></li>
</ul>
</li>
<li>GitHubのcopilotがVSCodeから可能に
<ul>
<li><a href="https://twitter.com/code/status/1682435342610079761?s=20">https://twitter.com/code/status/1682435342610079761?s=20</a></li>
</ul>
</li>
<li>TypeChat、マイクロソフトによる、プロンプトの代わりにType(型）をつかったChat、スキーマエンジニアリングともよぶらしい。
<ul>
<li><a href="https://github.com/microsoft/TypeChat">https://github.com/microsoft/TypeChat</a></li>
</ul>
</li>
<li>LLaMa2は、洞察とメタ認知に優れている
<ul>
<li><a href="https://arxiv.org/pdf/2307.10928.pdf">https://arxiv.org/pdf/2307.10928.pdf</a></li>
</ul>
</li>
<li>LangChainのLLaMa2インターフェイス
<ul>
<li><a href="https://python.langchain.com/docs/integrations/chat/llama_api">https://python.langchain.com/docs/integrations/chat/llama_api</a></li>
</ul>
</li>
<li>llama2 13B chat 4bit
<ul>
<li><a href="https://twitter.com/manjiroukeigo/status/1683047350141599744?s=20">https://twitter.com/manjiroukeigo/status/1683047350141599744?s=20</a></li>
</ul>
</li>
<li>京大、仏典をGPT-４で学習した、ブッダポッドプラスを発表
<ul>
<li><a href="https://ledge.ai/articles/buddha_bot_plus_kyoto_university">https://ledge.ai/articles/buddha_bot_plus_kyoto_university</a></li>
<li>GPT-4で仏典を解釈 わかりやすく回答</li>
</ul>
</li>
<li>TheBloke/Llama-2-70B-Chat-GGML
<ul>
<li><a href="https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML">https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML</a></li>
</ul>
</li>
<li>生成AIによるコード生成とCode Interpreter活用ハンズオン with PLATEAU
<ul>
<li><a href="https://connpass.com/event/290745/">https://connpass.com/event/290745/</a></li>
</ul>
</li>
<li>Abstraction and Analogy: The Keys to Robust Artificial Intelligence
<ul>
<li><a href="https://www.eventbrite.co.uk/e/abstraction-and-analogy-the-keys-to-robust-artificial-intelligence-tickets-675075728677?aff=oddtdtcreator">https://www.eventbrite.co.uk/e/abstraction-and-analogy-the-keys-to-robust-artificial-intelligence-tickets-675075728677?aff=oddtdtcreator</a></li>
</ul>
</li>
<li>MicrosoftによるOpenAI　Azure大全
<ul>
<li><a href="https://speakerdeck.com/hirosatogamo/chatgpt-azure-openai-da-quan">https://speakerdeck.com/hirosatogamo/chatgpt-azure-openai-da-quan</a></li>
<li>GPTの全体像、MicrosoftとOpenAIの関係、プロンプトエンジニアリングなど全て学べます</li>
</ul>
</li>
<li>llama2-webui
<ul>
<li><a href="https://github.com/liltom-eth/llama2-webui">https://github.com/liltom-eth/llama2-webui</a></li>
<li>Run Llama 2 locally with gradio UI on GPU or CPU from anywhere</li>
</ul>
</li>
<li>DeepMindから強化学習で核融合炉（トカマク）を制御する話
<ul>
<li><a href="https://arxiv.org/abs/2307.11546">https://arxiv.org/abs/2307.11546</a></li>
</ul>
</li>
<li>Google Colab で Llama 2 + LangChain の RetrievalQA を試す
<ul>
<li><a href="https://note.com/npaka/n/n6d33c2181050">https://note.com/npaka/n/n6d33c2181050</a></li>
</ul>
</li>
<li>医療のあらゆるタスクで最優秀スコアを獲得する医療特化の大規模言語モデル「Med-PaLM M」
<ul>
<li><a href="https://arxiv.org/pdf/2307.14334.pdf">https://arxiv.org/pdf/2307.14334.pdf</a></li>
</ul>
</li>
<li>「生成AIと著作権に関する論点整理」
<ul>
<li>なんと詳細な図が、、</li>
<li><a href="https://www.bunka.go.jp/seisaku/bunkashingikai/chosakuken/hoseido/r05_01/?fbclid=IwAR06f_2GFjUTlVn6Ofot52SfMhcJuyjTtkzF-D7DczgB75d0d5iCC9ucGnQ">https://www.bunka.go.jp/seisaku/bunkashingikai/chosakuken/hoseido/r05_01/?fbclid=IwAR06f_2GFjUTlVn6Ofot52SfMhcJuyjTtkzF-D7DczgB75d0d5iCC9ucGnQ</a></li>
</ul>
</li>
<li>World Coinの発表（Sam Altmanが関係している）、日本でも認証Orbが設置
<ul>
<li>代官山のサイトに行ってみたが、人はぼちぼち、日本では今一歩の認知度か。暑かった</li>
<li><a href="https://twitter.com/umiyuki_ai/status/1685323501069299713?s=20">https://twitter.com/umiyuki_ai/status/1685323501069299713?s=20</a></li>
<li>200万人がオーブ認証済みとか言ってたのに、予約者さえまだ32万人</li>
</ul>
</li>
</ul>
<h2 id="section-9">7/18</h2>
<p>暑くてすでに夏バテです。あいも変わらずcode interpreterの事例が続々、来年度の講義資料もこれで作るか。LLM時代のリテラシーって何という問い、教育もそうだし、リカレントもそう。Promptflowみたいな、（一見）思い付きのスタートアップがタケノコのように出てくるだろう。AlphaFoldがFoldItというゲームから名前がきているとは知らなかった、集合知ね。GoogleのNotebookLLM、エンジニアノートバッドという従来からの夢が、一歩実現に近づくか。普通に使っているEmbeddingなんかも、もちゃんと振り返って、カスタマイズの余地がある。</p>
<ul>
<li>ChatGPTのcode interpreterをつかて、講義の一部を作成（東大、強化学習、今井先生）
<ul>
<li><a href="https://twitter.com/ImAI_Eruel/status/1678378444441387010?s=20">https://twitter.com/ImAI_Eruel/status/1678378444441387010?s=20</a></li>
</ul>
</li>
<li>What Should Data Science Education Do with Large Language Models?
<ul>
<li><a href="https://arxiv.org/abs/2307.02792v2">https://arxiv.org/abs/2307.02792v2</a></li>
<li>LLMにより教育の変革、LLM-informed creativity, critical thinking, AI-guided programming.</li>
</ul>
</li>
<li>AlpacaEvalなるLLMベンチマークがあった、OSS系では、Vicuna-33Bがトップ
<ul>
<li><a href="https://tatsu-lab.github.io/alpaca_eval/">https://tatsu-lab.github.io/alpaca_eval/</a></li>
</ul>
</li>
<li>AI tools are designing entirely new proteins that could transform medicine
<ul>
<li><a href="https://www.nature.com/articles/d41586-023-02227-y">https://www.nature.com/articles/d41586-023-02227-y</a></li>
<li>RFdiffusionという拡散モデルによるタンパク質の合成が、AlphaFoldなどのハルシーネーションベース？の手法より優れているとの論文</li>
</ul>
</li>
<li>GPT-4でワークフロー自動化「Promptflow」開発、Carnot（カルノー）が8,500万円をプレシード調達
<ul>
<li><a href="https://thebridge.jp/2023/07/carnot-pre-seed-round-funding">https://thebridge.jp/2023/07/carnot-pre-seed-round-funding</a></li>
<li>雨後のタケノコのようにスタートアップが立ち上がるか？？</li>
</ul>
</li>
<li>LLamaindexにおける、RAGの説明 by npakaさん
<ul>
<li><a href="https://note.com/npaka/n/n27a36f784fb3">https://note.com/npaka/n/n27a36f784fb3</a></li>
<li>LLMとカスタムデータを組み合わせるための「RAG」(Retrieval Augmented Generation) パラダイム</li>
</ul>
</li>
<li>ストラング先生の線形代数講義のグラフィカルなノート、行列演算を極める。
<ul>
<li><a href="https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra">https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra</a></li>
</ul>
</li>
<li>DeepMindのHasabisさんのインタビュー
<ul>
<li>AlphaGoのあとにAlphaFoldに着手したのは、FoldIt（集合知で折り畳み問題を解くゲーム）に着想を得たとのこと</li>
<li><a href="https://podcasts.apple.com/us/podcast/a-i-could-solve-some-of-humanitys-hardest-problems/id1548604447?i=1000620748039">https://podcasts.apple.com/us/podcast/a-i-could-solve-some-of-humanitys-hardest-problems/id1548604447?i=1000620748039</a></li>
</ul>
</li>
<li>OpenAIから、embeddingのカスタマイズする、ノートブック。デフォルトでは使えない？
<ul>
<li><a href="https://github.com/openai/openai-cookbook/blob/main/examples/Customizing_embeddings.ipynb">https://github.com/openai/openai-cookbook/blob/main/examples/Customizing_embeddings.ipynb</a></li>
</ul>
</li>
<li>Google Labs、言語モデル「NotebookLM」の提供開始を発表–まず米国から
<ul>
<li><a href="https://japan.zdnet.com/article/35206577/">https://japan.zdnet.com/article/35206577/</a></li>
<li>NotebookLMではユーザーのノートや情報源を『土台にして』言語モデルが稼働</li>
</ul>
</li>
<li>Bardにマルチモーダル機能が。。
<ul>
<li><a href="https://twitter.com/i/status/1680237703676190722">https://twitter.com/i/status/1680237703676190722</a></li>
</ul>
</li>
<li></li>
</ul>
<h2 id="section-10">7/10</h2>
<p>OpenAIからGPT plusユーザー向けに、code interpreterが開放された。これで、データサイエンティストの仕事がなくなる？Plusじゃない人も、まずは手始めにLangChainのビデオをみて、データとのチャットを体感してみるといいかも。楔文字の翻訳など、様々な学問領域にLLMが侵食してゆく。OpenAIはアラインメント問題をAIで解くみたいなそっちの方向（結果としてAIの基盤整備が進む）。LLMのベンチマークとして、性格診断(Big5)というのは面白いアプローチ。使う人の性格判断と合わせるとマッチングが取れたりして。</p>
<ul>
<li>llamaindexにてtext-to-SQLの大幅なアップデート
<ul>
<li><a href="https://twitter.com/llama_index/status/1676002583381692421?s=20">https://twitter.com/llama_index/status/1676002583381692421?s=20</a></li>
</ul>
</li>
<li>大規模言語モデルの"性格"特性を分析＆調整するフレームワーク、DeepMind、ケンブリッジ大学、慶応大学
<ul>
<li>Personality Traits in Large Language Models</li>
<li><a href="https://arxiv.org/abs/2307.00184">https://arxiv.org/abs/2307.00184</a></li>
</ul>
</li>
<li>タスクの複雑さが増すとLLMの性能が急速に低下する現象を丁寧に検証
<ul>
<li>Faith and Fate: Limits of Transformers on Compositionality</li>
<li><a href="https://arxiv.org/abs/2305.18654">https://arxiv.org/abs/2305.18654</a></li>
</ul>
</li>
<li>LangChainにおけるSpacy Embeddingの利用例
<ul>
<li>OpenAIやHuggingFace以外、</li>
<li><a href="https://python.langchain.com/docs/modules/data_connection/text_embedding/integrations/spacy_embedding">https://python.langchain.com/docs/modules/data_connection/text_embedding/integrations/spacy_embedding</a></li>
</ul>
</li>
<li>EUのAI規制の、最終案の一つ前の和訳が、総務省の「AIネットワーク社会推進会議」で公開
<ul>
<li><a href="https://www.aplawjapan.com/publications/20220725">https://www.aplawjapan.com/publications/20220725</a></li>
</ul>
</li>
<li>IPAに「デジタル基盤センター」新設、デジタル庁と協力して基盤整備
<ul>
<li><a href="https://xtech.nikkei.com/atcl/nxt/news/18/15517/">https://xtech.nikkei.com/atcl/nxt/news/18/15517/</a></li>
<li>古巣の社会基盤センターが改組されて、「デジタル基盤センター」になり、デジタル庁の影響を受けるようになった。。。悲しい。</li>
</ul>
</li>
<li>CAMEL-5B と SentenceTransformers で LlamaIndex を試す
<ul>
<li><a href="https://note.com/npaka/n/n2e408cded4ac">https://note.com/npaka/n/n2e408cded4ac</a></li>
</ul>
</li>
<li>DeepLearningAIから、新コース、LangChain: Chat with Your Dataを無償リリース
<ul>
<li><a href="https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/">https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/</a></li>
</ul>
</li>
<li>OpenAI、計算リソース利用の２０％をアライメント問題に割くことを表明
<ul>
<li>アラインメント問題自体をAIで自動化するようにも見える（つまり目には目を、AIにはAIを）</li>
<li><a href="https://openai.com/blog/introducing-superalignment">https://openai.com/blog/introducing-superalignment</a></li>
</ul>
</li>
<li>NP困難といわれる、3次元パッキング問題を、MITが解く？
<ul>
<li><a href="https://news.mit.edu/2023/chore-packing-just-got-faster-and-easier-0706">https://news.mit.edu/2023/chore-packing-just-got-faster-and-easier-0706</a></li>
<li>FFTを利用しているとのこと</li>
</ul>
</li>
<li>楔形文字の解読にトランスフォーマを駆使して成功
<ul>
<li><a href="https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349?login=false">https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349?login=false</a></li>
</ul>
</li>
<li>OpenAI Code Interpreterを、GPT plusユーザーに解放。</li>
<li></li>
</ul>
<h2 id="section-11">7/4</h2>
<p>暑くてバテてました。LLMって、人間の知能を模擬するならば、Agentが実装できるというが、実装が近づいてきた。計画問題も直接解かせるよりも、計画問題を生成させるという組み合わせも面白い。形式言語なんか振り返ってみるのも面白いかも。LLMをComputer Visonへの応用、言語と画像の区別はなくなるのか？GoogleのKaggleチャレンジって、LLMの品質保証では重要な要素。ノン・セミパラメトリック統計ってのがあるのか？ 岡野原さんの『大規模言語モデルは新たな知能か』はおすすめ。記号接地って、LLMで実現できてんじゃない？みたいなのがじわじわと語られつつある(MLSE2023合宿より）。DeepMindのGemini、本当に出るのか？</p>
<ul>
<li>OpenAIのilian WengによるLLMをつかった、Agentの良解説記事
<ul>
<li><a href="https://lilianweng.github.io/posts/2023-06-23-agent/">https://lilianweng.github.io/posts/2023-06-23-agent/</a></li>
</ul>
</li>
<li>LLMを使ってプランニング問題を解く、PDDLと呼ばれるプランニング言語に変換させた上でソルバーに解かせる。LLM単独より正確。
<ul>
<li><a href="https://arxiv.org/abs/2304.11477">https://arxiv.org/abs/2304.11477</a></li>
</ul>
</li>
<li>Relicの社内勉強会での生成AI解説７０P資料
<ul>
<li><a href="https://qiita.com/hedgehog051/items/b1308e8baf7b0f551548">https://qiita.com/hedgehog051/items/b1308e8baf7b0f551548</a></li>
</ul>
</li>
<li>形式言語とは何か（現代思想）
<ul>
<li><a href="http://www.seidosha.co.jp/book/index.php?id=3821&amp;status=published">http://www.seidosha.co.jp/book/index.php?id=3821&amp;status=published</a></li>
<li>「正しい文とは何だろうか。。。」から始まる</li>
</ul>
</li>
<li>Towards Language Models That Can See: Computer Vision Through the LENS of Natural Language
<ul>
<li><a href="https://huggingface.co/papers/2306.16410">https://huggingface.co/papers/2306.16410</a></li>
</ul>
</li>
<li>学習済みモデルから特定のデータの影響を消すKaggleチャレンジ by Google
<ul>
<li><a href="https://ai.googleblog.com/2023/06/announcing-first-machine-unlearning.html">https://ai.googleblog.com/2023/06/announcing-first-machine-unlearning.html</a></li>
</ul>
</li>
<li>研究者の資質と教員の仕事 by 谷中教授
<ul>
<li><a href="https://twitter.com/verypluming/status/1674445457463062534?s=20">https://twitter.com/verypluming/status/1674445457463062534?s=20</a></li>
</ul>
</li>
<li>コンテキストを気にした文書分割
<ul>
<li><a href="https://twitter.com/RLanceMartin/status/1674817117475188737?s=20">https://twitter.com/RLanceMartin/status/1674817117475188737?s=20</a></li>
</ul>
</li>
<li>ノン・セミパラメトリック統計
<ul>
<li><a href="https://www.kyoritsu-pub.co.jp/book/b10031225.html">https://www.kyoritsu-pub.co.jp/book/b10031225.html</a></li>
<li>分布関数、密度関数や回帰関数について、一定の滑らかさのみを仮定して、ノンパラメトリックな推定と検定を行う方法を紹介する</li>
</ul>
</li>
<li>DeepMindの次世代AI「Gemini」はChatGPTを凌駕する？
<ul>
<li><a href="https://wired.jp/article/google-deepmind-demis-hassabis-chatgpt/?utm_medium=social&amp;utm_source=twitter">https://wired.jp/article/google-deepmind-demis-hassabis-chatgpt/?utm_medium=social&amp;utm_source=twitter</a></li>
<li>「GeminiはAlphaGoのようなシステムの強みと大規模言語モデルの卓越した言語能力を組み合わせたもの」</li>
</ul>
</li>
<li>岡野原『大規模言語モデルは新たな知能か』の個人的着目ポイントは、transformerで交互に積層する自己注意機構とパーセプトロンについて、前者が「短期記憶」、後者が長期記憶として機能するとの解説（p.108）</li>
<li>言語モデルに物理化学特徴量を取り入れた物性予測 by IBM
<ul>
<li>分子の物理化学的特徴量を選定し、言語モデルを微調整する</li>
<li><a href="https://arxiv.org/abs/2306.14919v1">https://arxiv.org/abs/2306.14919v1</a></li>
</ul>
</li>
<li>VARモデル ＋ グレンジャー因果性の統計的仮説検定による、時系列データの因果探索
<ul>
<li><a href="https://twitter.com/kenken26679105/status/1675281986917900288?s=20">https://twitter.com/kenken26679105/status/1675281986917900288?s=20</a></li>
<li>非ガウスモデル・VAR-LiNGAMであれば、同時刻も分析可能</li>
<li><a href="https://twitter.com/kenken26679105/status/1675306307849699328?s=20">https://twitter.com/kenken26679105/status/1675306307849699328?s=20</a></li>
</ul>
</li>
<li>Chains vs Agents" webinar by LangChain
<ul>
<li><a href="https://www.youtube.com/watch?v=bYLHklxEd_k">https://www.youtube.com/watch?v=bYLHklxEd_k</a></li>
</ul>
</li>
<li>結局記号接地ってなんだったけ？ by 丸山＠MLSE
<ul>
<li><a href="https://twitter.com/maruyama/status/1675813852947308544?s=20">https://twitter.com/maruyama/status/1675813852947308544?s=20</a></li>
<li>実世界への参照なしで、言語空間内での埋め込みだけで意味を操作</li>
</ul>
</li>
</ul>
<h2 id="section-12">6/26</h2>
<p>あいもかわらずOpenAIのFunction APIの利用について、具体例が増える、Pydatanicと組み合わせればほぼ無滝の情報抽出ができそうだし、抽出した情報をつかったQ&amp;Aなど、ちょっと説明性もあがるか？ヘルスケア分野でのGoogleAIの発表は衝撃的、眼科検診で様々な病気が見つかる。。。OpenLLAMaがでてきて、あっというまにFlanのデータでファインチューニングしたものが、商用利用できるのか？生成AIの研究や仕事への影響についてまとまった資料がぼちぼちでてきた。世界モデルに基づくプランニングなんかもLLMならではの研究か。LLMのコンパクト化も引き続き、マイクロソフトの取り組みがある。MITの試験問題をGPT4に解かせる話が不正という記事が、ちょっと悲しいが、オープンサイエンスの成果か。。</p>
<ul>
<li>OpenLLaMAは、LLaMaのオープン版（商用利用が可能？）GPU RAMは26.5GBで動作の模様
<ul>
<li><a href="https://huggingface.co/openlm-research/open_llama_13b">https://huggingface.co/openlm-research/open_llama_13b</a></li>
<li><a href="https://github.com/openlm-research/open_llama">https://github.com/openlm-research/open_llama</a></li>
</ul>
</li>
<li>Google	 ピーチャイ氏の講演、ヘルスケア分野で、AIがCTとかMRIとかを代替するかも（眼底検査で代替できる？）
<ul>
<li><a href="https://twitter.com/alvinfoo/status/1670599368930656257?s=20">https://twitter.com/alvinfoo/status/1670599368930656257?s=20</a></li>
</ul>
</li>
<li>OpenAIのFunction callとpydantic 	を組み合わせた例や再帰構造への対応など、情報抽出がこんなに便利に
<ul>
<li><a href="https://twitter.com/jxnlco/status/1670764386447953921?s=20">https://twitter.com/jxnlco/status/1670764386447953921?s=20</a></li>
<li><a href="https://twitter.com/matchaman11/status/1670799349004083200?s=20">https://twitter.com/matchaman11/status/1670799349004083200?s=20</a></li>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/output_parsing/openai_pydantic_program.html">https://gpt-index.readthedocs.io/en/latest/examples/output_parsing/openai_pydantic_program.html</a></li>
</ul>
</li>
<li>エンコーダーとデコーダについてわかりやすい解説
<ul>
<li><a href="https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder">https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder</a></li>
</ul>
</li>
<li>LangChainで、	<strong>MarkdownHeaderTextSplitter</strong>を使えば、引用元つきのQ&amp;Aが簡単に
<ul>
<li><a href="https://note.com/hamachi_jp/n/nf23b75d14068">https://note.com/hamachi_jp/n/nf23b75d14068</a></li>
</ul>
</li>
<li>マッキンゼーによる、生成AIの生産性への影響レポート
<ul>
<li>生成AIで従業員の時間を6~7割節約可能</li>
<li>生成AIのビジネスインパクトが高いのは2枚目画像の右上の領域</li>
<li>産業×用途別のインパクト評価(3枚目)</li>
<li>特定領域の具体的な用途と経済価値(4枚目)</li>
<li><a href="https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai-the-next-productivity-frontier#key-insights">https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai-the-next-productivity-frontier#key-insights</a></li>
</ul>
</li>
<li>GPT４ALLベースのcopilotが登場？
<ul>
<li><a href="https://morph.so/">https://morph.so/</a></li>
</ul>
</li>
<li>Debate　Tree、議論の構造をビジュアライズする
<ul>
<li><a href="https://debatetreeofthoughts.streamlit.app/">https://debatetreeofthoughts.streamlit.app/</a></li>
</ul>
</li>
<li>基盤モデル・生成AIの科学研究への影響に関する資料、文科省 基礎研究振興部会(第11回)
<ul>
<li><a href="https://www.mext.go.jp/b_menu/shingi/gijyutu/gijyutu27/siryo/mext_00007.html">https://www.mext.go.jp/b_menu/shingi/gijyutu/gijyutu27/siryo/mext_00007.html</a></li>
</ul>
</li>
<li>LlamaIndex	で、function call+pydatnicを組み合わせて、	query planningが可能に、
<ul>
<li><a href="https://gpt-index.readthedocs.io/en/latest/examples/agent/openai_agent_query_plan.html">https://gpt-index.readthedocs.io/en/latest/examples/agent/openai_agent_query_plan.html</a></li>
</ul>
</li>
<li>水口画伯なくなる、合掌
<ul>
<li><a href="https://twitter.com/AKZ161/status/1671498721287352320?s=20">https://twitter.com/AKZ161/status/1671498721287352320?s=20</a></li>
</ul>
</li>
<li>PyRCA、Pythonをつかった、ルート原因分析
<ul>
<li><a href="https://github.com/salesforce/PyRCA">https://github.com/salesforce/PyRCA</a></li>
</ul>
</li>
<li>OpenAIのFunction APIの解説
<ul>
<li><a href="https://every.to/chain-of-thought/gpt-4-can-use-tools-now-that-s-a-big-deal">https://every.to/chain-of-thought/gpt-4-can-use-tools-now-that-s-a-big-deal</a></li>
</ul>
</li>
<li>Flan-Open-Llama-7b、OpenLLaMaを、Flanのデータセットでチューニングした？
<ul>
<li><a href="https://huggingface.co/conceptofmind/Flan-Open-Llama-7b">https://huggingface.co/conceptofmind/Flan-Open-Llama-7b</a></li>
</ul>
</li>
<li>第2回LLM勉強会
<ul>
<li><a href="https://llm-jp.nii.ac.jp/llm/2023/06/20/study-group-2.html">https://llm-jp.nii.ac.jp/llm/2023/06/20/study-group-2.html</a></li>
</ul>
</li>
<li>local llmでsentence embeddingどれ使えば良いんだっけ
<ul>
<li><a href="https://note.com/if001/n/n25d795afe571">https://note.com/if001/n/n25d795afe571</a></li>
</ul>
</li>
<li>OpenAIのEmbbedingをつかって文章の類似度を計算
<ul>
<li><a href="https://techblog.gmo-ap.jp/2023/06/22/embeddings_api_calc_sentence_similarity/">https://techblog.gmo-ap.jp/2023/06/22/embeddings_api_calc_sentence_similarity/</a></li>
</ul>
</li>
<li>CVPR2023より、疑似確率が確率になるという問題への回答
<ul>
<li><a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Class_Adaptive_Network_Calibration_CVPR_2023_paper.pdf">https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Class_Adaptive_Network_Calibration_CVPR_2023_paper.pdf</a></li>
</ul>
</li>
<li>マイクロソフトから小規模LLMに関する論文 Textbooks Are All You Need
<ul>
<li>13億パラメータ"しか”ないモデル(phi-1)を、The StackとStackOverflowのデータを教科書品質にした60億トークンとGPT-3.5で生成した10億トークンをNVIDIA A100 8台・4日間で学習</li>
<li><a href="https://arxiv.org/abs/2306.11644">https://arxiv.org/abs/2306.11644</a></li>
</ul>
</li>
<li>Flan-Open-Llama-3b
<ul>
<li><a href="https://huggingface.co/conceptofmind/Flan-Open-Llama-3b">https://huggingface.co/conceptofmind/Flan-Open-Llama-3b</a></li>
</ul>
</li>
<li>Reasoning with Language Model is Planning with World Model
<ul>
<li>LLMを使ってプランニングを必要とするタスクを解く際、現在の状態をLLMを使って把握するようにして(「世界モデル」)、取るべき行動に対する報酬をLLMを使って見積もった上でモンテカルロ木探索によって行動を決定する手法(RAP; Reasoning via Planning)</li>
<li><a href="https://arxiv.org/abs/2305.14992">https://arxiv.org/abs/2305.14992</a></li>
</ul>
</li>
<li>OpenAIのCookbookにllama_indexをつかった、文書分析の例が載る
<ul>
<li><a href="https://github.com/openai/openai-cookbook/blob/main/examples/third_party_examples/financial_document_analysis_with_llamaindex.ipynb">https://github.com/openai/openai-cookbook/blob/main/examples/third_party_examples/financial_document_analysis_with_llamaindex.ipynb</a></li>
</ul>
</li>
<li>GPT-4がMITの試験問題を正しく解いたという論文が手続き的にも本質的内容面でも不正との指摘
<ul>
<li>正解がでるまで何度も聞いた等の不正があった模様。。</li>
<li><a href="http://people.csail.mit.edu/asolar/CoursesPaperStatement.pdf">http://people.csail.mit.edu/asolar/CoursesPaperStatement.pdf</a></li>
</ul>
</li>
</ul>
<h2 id="section-13">6/19</h2>
<p>今週は、6/12日にCEOの慶応大学での講演。OpenAIのAPIのアップデートの話題がありました。トークン数の拡大（青空文庫の短編クラスなら取り扱える）とか、Function callの追加。これで、LangChainのReActエージェントを使わなくても、OpenAI Agentで外部ツールとLLMが連携したソリューションが手軽に作れるようになるのは恐ろしいこと。品質評価の視点では、OpenAI Evalとかの話題も。欧州ＡＩ規制が、欧州議会の投票で採択され、次の段階（トリローグ）を経て年内に成立か。さっそくスタンフォード大学のHAIチームが、既存LLMの規制への対応状況をベンチマーク。データサイエンス系の型は、関数データ解析などは目からうろこではないか。ベイズ派と頻度派の争いには巻き込まれたくないもの。知識グラフとLLMの融合も、整理された論文が出てきた。</p>
<ul>
<li>「OpenAI CEO Sam Altman氏と塾生との対話」開催(6/12)
<ul>
<li><a href="https://www.keio.ac.jp/ja/news/2023/6/15/27-139184/">https://www.keio.ac.jp/ja/news/2023/6/15/27-139184/</a></li>
<li>会場となった西校舎ホールには約700名の学生が集まり、約40分にわたり活発な質疑応答が行われました。またその様子は519教室にも配信され、1,000名以上の学生にとって貴重な機会となりました。</li>
</ul>
</li>
<li>「二つの分散:不偏推定量と最尤推定量のどち らを使うべきか」井手さん(IBM)
<ul>
<li>頻度派に対する執拗な攻撃は続く。。</li>
<li><a href="https://ide-research.net/book/Which_variance_should_I_use.pdf">https://ide-research.net/book/Which_variance_should_I_use.pdf</a></li>
<li>データサイエンスの本質をひと言で答えろと言われたら、「観 測データに対してもっとも当てはまりの良いモデルをつくるために、最尤推 定を使ってパラメターを決めること」と答えればよい」</li>
</ul>
</li>
<li>Evaluating the Social Impact of Generative AI Systems in Systems and Society
<ul>
<li><a href="https://huggingface.co/papers/2306.05949">https://huggingface.co/papers/2306.05949</a></li>
</ul>
</li>
<li>「見たくないものをみる」（PFNの丸山さん）
<ul>
<li><a href="https://note.com/hiroshi_maruyama/n/n7890a1fb7aef">https://note.com/hiroshi_maruyama/n/n7890a1fb7aef</a></li>
<li>新たな倫理規範の確立について</li>
<li>「人間中心のAI」に違和感を抱き、「人間が（知能の面で）万物の霊長でないかもしれない」という「都合の悪い真実」を直視すべきという話</li>
</ul>
</li>
<li>「デジタル庁のサイトやばすぎる」
<ul>
<li><a href="https://qiita.com/mu_tomoya/items/f78f1fad3a8b57ac7dc3">https://qiita.com/mu_tomoya/items/f78f1fad3a8b57ac7dc3</a></li>
<li>やばいくらい参考になるらしい。</li>
</ul>
</li>
<li>米OpenAIのCEO「AIはさらに賢く」　慶大で意見交換（日経）
<ul>
<li>OpenAIの強みはresearch culture</li>
<li>コーディング or 理論解析に強い人が成功している</li>
<li>AI技術は急速に進歩・応用されており、このような時代にAIに関われる今の学生はlucky generation</li>
<li><a href="https://www.nikkei.com/article/DGXZQOUC1037N0Q3A610C2000000/">https://www.nikkei.com/article/DGXZQOUC1037N0Q3A610C2000000/</a></li>
</ul>
</li>
<li>平均・分散・相関が変わらない、X,Yの様々な事例。。
<ul>
<li>まあ有名な奴だけどゴジラはよく考えたな。</li>
<li><a href="https://twitter.com/docmilanfar/status/1668093023895568386?s=20">https://twitter.com/docmilanfar/status/1668093023895568386?s=20</a></li>
</ul>
</li>
<li>ヒントン先生にたいするルカンの所感
<ul>
<li>人間並みのAIを実現するには、２つが必須で、（今はたらない）
<ul>
<li>(1) learning world models from sensory inputs like video,</li>
<li>(2) an architecture that can reason and plan (not just auto-regress).</li>
</ul>
</li>
</ul>
</li>
<li>Dockerコンテナをwebassemblyに変換して実行できるツール？
<ul>
<li><a href="https://www.publickey1.jp/blog/23/dockerwebassemblywebcontainer2wasm03.html">https://www.publickey1.jp/blog/23/dockerwebassemblywebcontainer2wasm03.html</a></li>
</ul>
</li>
<li>欧州のデータスペースに関する、JRCのレポート
<ul>
<li>European Data Spaces - Scientific Insights into Data Sharing and Utilisation at Scale</li>
<li><a href="https://publications.jrc.ec.europa.eu/repository/handle/JRC129900">https://publications.jrc.ec.europa.eu/repository/handle/JRC129900</a></li>
</ul>
</li>
<li>GPTにFunction Callが追加
<ul>
<li>出力の整形とか、あるいは、自然言語から、関数のAPI呼び出しを作ったりとか、つまり、LangChainでいうところのAgentが簡単にできるようになる。。</li>
<li><a href="https://openai.com/blog/function-calling-and-other-api-updates">https://openai.com/blog/function-calling-and-other-api-updates</a></li>
</ul>
</li>
<li>CV（コンピュータビジョン）の最新刊は、生成AI、巻頭言がよかったとのこと
<ul>
<li><a href="https://www.amazon.co.jp/dp/B0C6JW6T6B?ref_=cm_sw_r_cp_ud_dp_Q44X6Q8W7NPXKP46168A">https://www.amazon.co.jp/dp/B0C6JW6T6B?ref_=cm_sw_r_cp_ud_dp_Q44X6Q8W7NPXKP46168A</a></li>
<li>イマドキノ拡散モデル：拡散モデルに関する最近の研究動向を紹介。基本技術、条件付き生成への拡張、生成の高速化について述べ、拡散モデルを学ぶうえで役立つリソースを紹介。</li>
</ul>
</li>
<li>OpenAIのモデルを評価するフレームワークEval
<ul>
<li><a href="https://github.com/openai/evals">https://github.com/openai/evals</a></li>
<li><strong>特定の課題に対してどれぐらい高精度で生成できているかを評価</strong>できます。</li>
</ul>
</li>
<li>DADCのスマートビルガイドラインの補足資料が公開、
<ul>
<li>補足資料って、最初から説明が足らなかっただろうに。</li>
<li><a href="https://www.ipa.go.jp/digital/architecture/Individual-link/ps6vr7000001x8o0-att/smartbuilding_guideline_appendix.pdf">https://www.ipa.go.jp/digital/architecture/Individual-link/ps6vr7000001x8o0-att/smartbuilding_guideline_appendix.pdf</a></li>
</ul>
</li>
<li>最近引退されたMITのストラング先生の、"Ther Art of Linear Algebra"の和訳、たった１４P、全理系は涙して読むべし
<ul>
<li>「行列5分解」「行列の世界」「固有値地図」の視覚的解説</li>
<li><a href="https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra/blob/main/The-Art-of-Linear-Algebra-j.pdf">https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra/blob/main/The-Art-of-Linear-Algebra-j.pdf</a></li>
</ul>
</li>
<li>GTP-calls:コールセンターの会話を分析するアプリ、マイクロソフト
<ul>
<li><a href="https://arxiv.org/abs/2306.07941">https://arxiv.org/abs/2306.07941</a></li>
</ul>
</li>
<li>シンボリック回帰と深層学習を組み合わせることで、データから方程式を見つける。
<ul>
<li><a href="https://arxiv.org/abs/2207.00529">https://arxiv.org/abs/2207.00529</a></li>
</ul>
</li>
<li>GPT3.5 APIのアプデで使えるようになった16kトークンで何ができるか？
<ul>
<li>青空文庫のちょっとした短編ならば、分析が可能になったレベルらしい</li>
<li><a href="https://note.com/mahlab/n/n99577fabf16e">https://note.com/mahlab/n/n99577fabf16e</a></li>
</ul>
</li>
<li>GPTでのfunction callの良例
<ul>
<li><a href="https://gist.github.com/hotchpotch/364cb8ae188e40f4e9ff1273232bc918">https://gist.github.com/hotchpotch/364cb8ae188e40f4e9ff1273232bc918</a></li>
</ul>
</li>
<li>OpenAI API の 関数呼び出し を試す、npakaさんの記事
<ul>
<li><strong>外部APIを呼び出して質問に答えるチャットボットの作成</strong></li>
<li><strong>自然言語をAPI呼び出しに変換</strong></li>
<li><strong>テキストから構造化データを抽出</strong></li>
<li><a href="https://note.com/npaka/n/n917463f55b8a">https://note.com/npaka/n/n917463f55b8a</a></li>
</ul>
</li>
<li>欧州AI規制における、生成モデル、一般目的AIに対する義務事項
<ul>
<li><a href="https://www.europarl.europa.eu/news/en/press-room/20230609IPR96212/meps-ready-to-negotiate-first-ever-rules-for-safe-and-transparent-ai">https://www.europarl.europa.eu/news/en/press-room/20230609IPR96212/meps-ready-to-negotiate-first-ever-rules-for-safe-and-transparent-ai</a></li>
<li>ban on AI for biometric surveillance, emotion recognition, predictive policing</li>
<li>registration of models with EU</li>
<li>detailed summary of training data</li>
<li>requirement to identify deepfakes</li>
</ul>
</li>
<li>第6回LangChainもくもく会開催レポート
<ul>
<li><a href="https://note.com/mahlab/n/nc6ec4a9bd3c5">https://note.com/mahlab/n/nc6ec4a9bd3c5</a></li>
<li>Grounded GenerationサービスVectara、LLMホスティングサービスBeam、SQLiteでベクトルDB検索が可能になるsqlite-vss、PostgreSQLのベクトル検索拡張pgvector、LangChain AI Handbookの話</li>
</ul>
</li>
<li>OpenAIのFunction Callをつかうと、LangChainのReAct Agentのようなこともできるという話（すげー、というか、エコシステム壊してないか？？？）
<ul>
<li><a href="https://github.com/jerryjliu/llama_index/blob/main/docs/examples/agent/openai_agent.ipynb">https://github.com/jerryjliu/llama_index/blob/main/docs/examples/agent/openai_agent.ipynb</a></li>
<li>llma_indexには、openai agentが組み込まれる予定らしい。</li>
<li><a href="https://twitter.com/llama_index/status/1668995628146257921?s=20">https://twitter.com/llama_index/status/1668995628146257921?s=20</a></li>
</ul>
</li>
<li>「関数データ解析の概要とその方法」滋賀大学、松井先生
<ul>
<li><a href="https://speakerdeck.com/hidetoshimatsui/guan-shu-detajie-xi-nogai-yao-tosonofang-fa">https://speakerdeck.com/hidetoshimatsui/guan-shu-detajie-xi-nogai-yao-tosonofang-fa</a></li>
<li>デーサイエンスで習う、回帰、クラスタリング、などのすべてが、データを関数として取り扱う枠組みで、再構成されている。なんともすがすがしいスライド。夏休みのお供に！</li>
</ul>
</li>
<li>機械学習サービスにおけるONNXの活用と応用　〜ONNXテキスト形式の拡張〜
<ul>
<li><a href="https://www.sportip.jp/blogs/onnx">https://www.sportip.jp/blogs/onnx</a></li>
<li>やっぱり、ONNXにして、WebGPUつかって、ブラウザで動か用になるのね、</li>
</ul>
</li>
<li>Rinna-3.6B で 文脈付きの質問応答 を試す npakaさん記事より
<ul>
<li><a href="https://note.com/npaka/n/n3bb60c61ef94">https://note.com/npaka/n/n3bb60c61ef94</a></li>
<li>「JSQuAD」は文脈付きの質問応答タスクで、53.42と半分以上正解</li>
</ul>
</li>
<li>欧州AI規制に、現状のLLMはどれぐらい対応できているかのベンチマーク(スタンフォード題）
<ul>
<li><a href="https://crfm.stanford.edu/2023/06/15/eu-ai-act.html">https://crfm.stanford.edu/2023/06/15/eu-ai-act.html</a></li>
<li>現状特に著作権保護学習データ開示等が行われていないこと、DSA的透明性確保の非対称規制提言など</li>
<li>すごすぎでしょう。</li>
</ul>
</li>
<li>知識グラフのLLMの統合についてのロードマップ論文
<ul>
<li>Unifying Large Language Models and Knowledge Graphs: A Roadmap</li>
<li><a href="https://arxiv.org/abs/2306.08302">https://arxiv.org/abs/2306.08302</a></li>
<li>Combining the advantages of LLMs and knowledge graphs (KGs) is a promising direction.</li>
</ul>
</li>
<li>欧州でEV電池規制　リチウムは8割再資源化、31年までに
<ul>
<li>6/14日に欧州議会の投票を通過したという話、</li>
<li>EV,主要材料のリチウムは使用済み電池から2027年までに50%、31年までに80%を再資源化する必要がある。</li>
<li>「電池パスポート」の導入も決まった</li>
<li><a href="https://www.nikkei.com/article/DGXZQOGR1706S0X10C23A6000000/">https://www.nikkei.com/article/DGXZQOGR1706S0X10C23A6000000/</a></li>
</ul>
</li>
<li>レヴィ＝ストロースの70年来の謎を進化シミュレーションで解明- 文化人類学の基礎「親族の構造」を数理モデルで生成 -
<ul>
<li><a href="https://www.u-tokyo.ac.jp/focus/ja/press/z0109_00325.html">https://www.u-tokyo.ac.jp/focus/ja/press/z0109_00325.html</a></li>
<li>これって、LLMで同じことが多分1年以内のできるようになる。</li>
</ul>
</li>
<li>応用行動分析「死人テスト」死人にもできることを行動目標にしたいという話
<ul>
<li><a href="https://twitter.com/81I6VVboj7h2Bqy/status/1667893285883621376?s=20">https://twitter.com/81I6VVboj7h2Bqy/status/1667893285883621376?s=20</a></li>
<li>「会議で余計な発言しない」、「廊下で走らない」、などは死人にもできる目標なので、それはまちがえであるということ。。その行動、死人にもできるのでは？</li>
</ul>
</li>
</ul>
<h2 id="section-14">6/12</h2>
<p>熊本で開かれた人工知能学会全国大会の話題もちらほら。偉い先生のまとめスライドが役に立つ。ローカルでLLMを動かす動きも相も変わらず活発。ggML形式のLLMならば、gpt4allのチャット用のソフトでllmを入れ替えて動くらしい。npaka氏のローカルＬＬＭのまとめは良記事。それにしても、4.75bitのSpQRって本当か？タンパク質やプロプロテインなどの研究対象の操作などができるチャットシステムも登場、そういう応用はこれからもたくさんでそう。データサイエンス界隈は、自らの存在意義的に、Noteableプラグインがよほど応えたらしい。ついにMSからChatでOffice製品を制御できる技術が発表、パワポも作ってくれるのか？その間googleのBardは、裏でコード実行する仕組みを取り入れ、苦手な計算とか論理などの精度が向上。今度はＧＡＳとの連携か。東芝福本氏の製造業における生成ＡＩの活用は一読の価値あり。倫理とか公平性という、上から目線より、「卵のためのＡＩ」に、わたしはなりたい。</p>
<ul>
<li>データ分析の効率が10倍上がるデータサイエンティストのためのChatGPTの活用術
<ul>
<li><a href="https://qiita.com/ot12/items/96b5783568196d3320fe">https://qiita.com/ot12/items/96b5783568196d3320fe</a></li>
<li>さいごはNoteableなのか。。</li>
</ul>
</li>
<li>ChatGPTのように狙いの分子やタンパク質を編集できるChatDrug
<ul>
<li><a href="https://arxiv.org/abs/2305.18090v1">https://arxiv.org/abs/2305.18090v1</a></li>
</ul>
</li>
<li>「rinna」の日本語言語モデルを試用、メモリ32GBあればCPUだけでも動くぞ！
<ul>
<li><a href="https://internet.watch.impress.co.jp/docs/column/shimizu/1503707.html">https://internet.watch.impress.co.jp/docs/column/shimizu/1503707.html</a></li>
</ul>
</li>
<li>GPT4ALL周りのソフトは、ggML準拠のモデルならば、gpt4allでなくても動くようになった！
<ul>
<li>The GPT4All Chat UI supports models from all newer versions of <code>ggML</code>, <code>llama.cpp</code> including the <code>LLaMA</code>, <code>MPT</code> and <code>GPT-J</code> architectures. T</li>
<li><a href="https://docs.gpt4all.io/gpt4all_chat.html">https://docs.gpt4all.io/gpt4all_chat.html</a></li>
</ul>
</li>
<li>どうやらパラメータ数130億(13B)でChatGPT(GPT-3.5)クラスの性能が出せることがMSから発表
<ul>
<li><a href="https://huggingface.co/papers/2306.02707">https://huggingface.co/papers/2306.02707</a></li>
</ul>
</li>
<li>こんどはProteinChat、構造があれば何でもよいのか。。
<ul>
<li>ProteinChat: Towards Achieving ChatGPT-Like Functionalities on Protein 3D Structures</li>
<li><a href="https://www.techrxiv.org/articles/preprint/ProteinChat_Towards_Achieving_ChatGPT-Like_Functionalities_on_Protein_3D_Structures/23120606/1">https://www.techrxiv.org/articles/preprint/ProteinChat_Towards_Achieving_ChatGPT-Like_Functionalities_on_Protein_3D_Structures/23120606/1</a></li>
</ul>
</li>
<li>確率的熱力学に経済学のツールを用いることで、熱力学と情報理論の間の相互作用について定量的に調べた
<ul>
<li><a href="https://arxiv.org/abs/2306.00449">https://arxiv.org/abs/2306.00449</a></li>
</ul>
</li>
<li>「大規模言語モデル入門」７月２９日発売予定
<ul>
<li><a href="https://www.amazon.co.jp/dp/4297136333">https://www.amazon.co.jp/dp/4297136333</a></li>
</ul>
</li>
<li>Microsoftの研究者らが新たに開発したAIシステム「Semantic Interpreter」は、Officeを操作、パワポが作れる。。
<ul>
<li><a href="https://arxiv.org/abs/2306.03460">https://arxiv.org/abs/2306.03460</a></li>
</ul>
</li>
<li>DeepMindのAlphaDev、人の作りしソートアルゴリズムよりも高速なアルゴリズムを生成。
<ul>
<li><a href="https://www.nature.com/articles/s41586-023-06004-9">https://www.nature.com/articles/s41586-023-06004-9</a></li>
<li>といっても最適化しているだけだとか、ChatGPTでも同様な最適化ができたとの報告が続く。</li>
<li><a href="https://chat.openai.com/share/95693df4-36cd-4241-9cae-2173e8fb760c">https://chat.openai.com/share/95693df4-36cd-4241-9cae-2173e8fb760c</a></li>
</ul>
</li>
<li>医療現場での、構造化されてない医療メモをつかったLLM
<ul>
<li><a href="https://www.nature.com/articles/s41586-023-06160-y">https://www.nature.com/articles/s41586-023-06160-y</a></li>
</ul>
</li>
<li>LlamaIndexの、JSON Query Engineの紹介ビデオ
<ul>
<li><a href="https://www.youtube.com/watch?v=4tDyfAaIqEw">https://www.youtube.com/watch?v=4tDyfAaIqEw</a></li>
</ul>
</li>
<li>前篇　AIは「ジェスチャーゲーム」を知らない
<ul>
<li>今井むつみ先生と、高野秀行の対談</li>
<li>『言語の本質　ことばはどう生まれ、進化したか』の今井先生の対談</li>
<li><a href="https://kangaeruhito.jp/interview/756531">https://kangaeruhito.jp/interview/756531</a></li>
</ul>
</li>
<li>Googleの「Bard」が「暗黙的なコード実行」を導入、文字列の操作や論理・推論を含む複雑なタスクに対する回答精度が向上
<ul>
<li>やっぱBardやるね。</li>
<li><a href="https://gigazine.net/news/20230608-google-bard-implicit-code-execution/">https://gigazine.net/news/20230608-google-bard-implicit-code-execution/</a></li>
</ul>
</li>
<li>Large Language Models are In-Context Semantic Reasoners rather than Symbolic Reasoners
<ul>
<li><a href="https://arxiv.org/abs/2305.14825">https://arxiv.org/abs/2305.14825</a></li>
</ul>
</li>
<li>東芝福本氏による、製造業で生成AIはどんな役割を果たすのか？ ドイツで見たMSやシーメンスらの取り組み
<ul>
<li>ハノーバーメッセでの展示の現地報告は貴重だ</li>
<li><a href="https://www.sbbit.jp/article/st/115632">https://www.sbbit.jp/article/st/115632</a></li>
</ul>
</li>
<li>壁のためのAIと卵のためのAI
<ul>
<li>JSAI2023での学生企画、卵のためのAIになりたい。</li>
<li><a href="https://speakerdeck.com/yukinobaba/ai-for-wall-and-ai-for-egg">https://speakerdeck.com/yukinobaba/ai-for-wall-and-ai-for-egg</a></li>
</ul>
</li>
<li>松尾研による、「基盤モデルの技術と展望」
<ul>
<li>GPT-3など基盤モデルの技術的な動向について、数多くの文献をもとに整理されている</li>
<li><a href="https://speakerdeck.com/yusuke0519/jsai2023-tutorial-ji-pan-moderunoji-shu-tozhan-wang">https://speakerdeck.com/yusuke0519/jsai2023-tutorial-ji-pan-moderunoji-shu-tozhan-wang</a></li>
</ul>
</li>
<li>状態空間モデルを活用した時系列データのCausalImpact分析
<ul>
<li><a href="https://speakerdeck.com/stakaya/zhuang-tai-kong-jian-moderuwohuo-yong-sita-shi-xi-lie-detafalsecausalimpactfen-xi">https://speakerdeck.com/stakaya/zhuang-tai-kong-jian-moderuwohuo-yong-sita-shi-xi-lie-detafalsecausalimpactfen-xi</a></li>
<li>まあ、Rでなくても再現できそうだ。</li>
</ul>
</li>
<li>llamaindexのKnowledge Graphインデクス機能が大幅にパワーアップ？</li>
<li><a href="https://github.com/jerryjliu/llama_index/blob/main/docs/examples/index_structs/knowledge_graph/NebulaGraphKGIndexDemo.ipynb">https://github.com/jerryjliu/llama_index/blob/main/docs/examples/index_structs/knowledge_graph/NebulaGraphKGIndexDemo.ipynb</a></li>
<li>欠損値を平均値を代入するというプラクティスが議論に、
<ul>
<li>問題ないという人もいるが、問題ないならそもそも除外しても同じなのでは？</li>
<li><a href="https://twitter.com/kenken26679105/status/1667288891949453312?s=20">https://twitter.com/kenken26679105/status/1667288891949453312?s=20</a></li>
</ul>
</li>
<li>JSAI2023、人の位置情報の時系列をトークン列に置き換えてGPT-2で学習、人の移動軌道を生成する研究
<ul>
<li><a href="https://confit.atlas.jp/guide/event/jsai2023/subject/2H5-OS-8a-02/tables?cryptoId=">https://confit.atlas.jp/guide/event/jsai2023/subject/2H5-OS-8a-02/tables?cryptoId=</a></li>
</ul>
</li>
<li>ローカルでlangchain経由で簡単につかえてそこそこ日本語も喋れるのwizard-vicuna-13 q8_0
<ul>
<li><a href="https://twitter.com/if_004/status/1667474091564204033?s=20">https://twitter.com/if_004/status/1667474091564204033?s=20</a></li>
</ul>
</li>
<li>ローカルLLMのまとめ
<ul>
<li><a href="https://note.com/npaka/n/nd95fba328b65">https://note.com/npaka/n/nd95fba328b65</a></li>
</ul>
</li>
<li>表形式データの行を1文と見て，差分プライベートに言語モデルを学習させ，そこから合成データを生成する手法を提案．複数のデータセットで既存のグラフィカルモデルベースのものと同等の性能
<ul>
<li><a href="https://arxiv.org/abs/2306.04803">https://arxiv.org/abs/2306.04803</a></li>
</ul>
</li>
<li>4.75bit 相当の量子化で、16fp と比べ損失ゼロの推論が可能
<ul>
<li>SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression</li>
<li><a href="https://arxiv.org/abs/2306.03078">https://arxiv.org/abs/2306.03078</a></li>
</ul>
</li>
</ul>
<h2 id="section-15">6/5</h2>
<p>相も変わらず、4bit化とか、Rinna-3.6BのLoRaとか、ローカルでLLMを動かす、作る可能性が広がっている。まあ現状のLLMって実は疎なのではという特異値分解の結果も。じゃらんでChatGPT活用サービス試行開始。DeepLearningAIより、LangChainのショートコース無料開始、作者登場で豪華なことに。DeepAI OpenAIのLLMの苦手な計算問題での、「プロセス監視報酬モデル(PRM)」による改良。数値シミュレーションの世界でもLLM活用が。OpenAIのsecurity Portal発表。Grokking「過学習してしばらく経ってから、急に汎化誤差が下がり始める（正解率が上がり始める）」という現象への手がかりも。。言語学会からもLLMに対する元気のよい発信や出版が多数。ドイツ連邦データ保護当局（BfDI）の生成AIについての声明、これは読むべきか。日本のAI戦略会議の議論との温度差はいかんともしがたい。新聞記事の本論の前段の記事間違えを鬼の首を取ったように叩く、狭量な日本のDSメンタリティも興味深い。この政府にしてこの国民アリという感じか、逆か。</p>
<ul>
<li>DeepLearningAIより、LangChainのショートコースが、無料
<ul>
<li>作者自身の登場で、豪華な構成に、</li>
<li>LangChain for LLM Application Development</li>
<li><a href="https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/">https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/</a></li>
</ul>
</li>
<li>biomedGPT: マルチモーダルな医療用のGPT
<ul>
<li><a href="https://arxiv.org/abs/2305.17100">https://arxiv.org/abs/2305.17100</a></li>
</ul>
</li>
<li>AI戦略会議による、暫定的論点整理、日本はノーローなのか？これでいいの？
<ul>
<li><a href="https://www8.cao.go.jp/cstp/ai/index.html">https://www8.cao.go.jp/cstp/ai/index.html</a></li>
</ul>
</li>
<li>LLM自身がPythonによるツール作成？Large Language Models as Tool Makers
<ul>
<li><a href="https://arxiv.org/abs/2305.17126">https://arxiv.org/abs/2305.17126</a></li>
</ul>
</li>
<li>Google Colabで、ローカルランタイムでの実行ができるようになった。。
<ul>
<li><a href="https://research.google.com/colaboratory/local-runtimes.html">https://research.google.com/colaboratory/local-runtimes.html</a></li>
</ul>
</li>
<li>Transcendental Style in Film(映画における超越的様式)
<ul>
<li><a href="https://twitter.com/routemopsy/status/1663396967417024513?s=20">https://twitter.com/routemopsy/status/1663396967417024513?s=20</a></li>
<li>ホウ・シャオシェンがタルコフスキー領域にあるのは解せない。</li>
</ul>
</li>
<li>「 Google Colab で Rinna-3.6B のLoRAファインチューニングを試す」
<ul>
<li>なんと14Gでできるなら、無料枠？</li>
<li><a href="https://note.com/npaka/n/nc387b639e50e">https://note.com/npaka/n/nc387b639e50e</a></li>
</ul>
</li>
<li>Large Language Models are not Fair Evaluators
<ul>
<li><a href="https://arxiv.org/pdf/2305.17926.pdf">https://arxiv.org/pdf/2305.17926.pdf</a></li>
</ul>
</li>
<li>How To Finetune GPT Like Large Language Models on a Custom Dataset
<ul>
<li>Macbookでも簡単にfinetuneできるようになった</li>
<li><a href="https://lightning.ai/pages/blog/how-to-finetune-gpt-like-large-language-models-on-a-custom-dataset/">https://lightning.ai/pages/blog/how-to-finetune-gpt-like-large-language-models-on-a-custom-dataset/</a></li>
</ul>
</li>
<li>「ChatGPTの仕組みと社会への影響」、京大黒橋先生のわかりやすいといわれる講義、１９分でさくっと
<ul>
<li><a href="https://www.youtube.com/watch?v=aKqIPlDyWhs">https://www.youtube.com/watch?v=aKqIPlDyWhs</a></li>
</ul>
</li>
<li>「ChatGPTとNoteableによる科学技術情報分析」
<ul>
<li><a href="https://speakerdeck.com/hayataka88/chatgpttonoteableniyoruke-xue-ji-shu-qing-bao-fen-xi">https://speakerdeck.com/hayataka88/chatgpttonoteableniyoruke-xue-ji-shu-qing-bao-fen-xi</a></li>
<li>噂のNoteable。ついに、お話しするだけで、EDAから回帰まで、、</li>
</ul>
</li>
<li>Let’s Verify Step by Step by OpenAI
<ul>
<li>LLMが苦手な計算問題をとかせるために、process supervisionというのをどうにゅ</li>
<li>「プロセス監視報酬モデル(PRM)」というらしい。</li>
<li><a href="https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf">https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf</a></li>
</ul>
</li>
<li>NIIのオープンハウス(6/1-6/3)、ChatGPTネタ大杉。
<ul>
<li><a href="https://www.nii.ac.jp/event/openhouse/2023/">https://www.nii.ac.jp/event/openhouse/2023/</a></li>
</ul>
</li>
<li>因果推論のコースマテリアル
<ul>
<li><a href="https://arxiv.org/abs/2305.18793">https://arxiv.org/abs/2305.18793</a></li>
</ul>
</li>
<li>Rinnaすごいかも。japanese-gpt-neox-3.6b-instruction-ppo
<ul>
<li><a href="https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo">https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo</a></li>
</ul>
</li>
<li>局所詳細釣り合い、ゆらぎの定理、Jarzynski等式と拡散モデルの関係
<ul>
<li><a href="https://zenn.dev/xiangze/articles/6e8ce8b8d43d08">https://zenn.dev/xiangze/articles/6e8ce8b8d43d08</a></li>
<li>そういうものらしい</li>
</ul>
</li>
<li>rinna/japanese-gpt-neox-3.6b について、ベース、SFT、RLHFで動かした例 on colab
<ul>
<li><a href="https://note.com/npaka/n/ne4a38239f420">https://note.com/npaka/n/ne4a38239f420</a></li>
<li>ベースなら無料枠で動く？</li>
</ul>
</li>
<li>じゃらんでAIチャットサービス開始
<ul>
<li><a href="https://note.com/npaka/n/ne4a38239f420">https://note.com/npaka/n/ne4a38239f420</a></li>
</ul>
</li>
<li>Rinna-3.6B を llama.cpp で CPU 動作のメモ
<ul>
<li><a href="https://zenn.dev/syoyo/articles/946c17666e10fb">https://zenn.dev/syoyo/articles/946c17666e10fb</a></li>
<li>CPUだけでも十分動くのか。。。</li>
</ul>
</li>
<li>Berry: A code for the differentiation of Bloch wavefunctions from DFT calculations
<ul>
<li><a href="https://arxiv.org/abs/2006.02744">https://arxiv.org/abs/2006.02744</a></li>
<li>DFT計算で得た波動関数を微分するためのオープンソース</li>
</ul>
</li>
<li>What Is ChatGPT Doing … and Why Does It Work?
<ul>
<li><a href="https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/">https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/</a></li>
<li>Wolfman AlphaのWolfmanさんの記事、</li>
<li>「現在のChatGPTの場合，事態はもっと極端で，各トークンの出力を生成するためのニューラルネットはループのない純粋な「フィードフォワード」ネットワークであるため，自明でない「制御フロー」を持ついかなる計算も行うことができない</li>
</ul>
</li>
<li>Googleによる図式を理解するLLM
<ul>
<li>Foundation models for reasoning on charts</li>
<li><a href="https://ai.googleblog.com/2023/05/foundation-models-for-reasoning-on.html">https://ai.googleblog.com/2023/05/foundation-models-for-reasoning-on.html</a></li>
</ul>
</li>
<li>Physics-constrained machine learning for scientific computing
<ul>
<li><a href="https://www.amazon.science/blog/physics-constrained-machine-learning-for-scientific-computing?_amp=true">https://www.amazon.science/blog/physics-constrained-machine-learning-for-scientific-computing?_amp=true</a></li>
<li>保存則と境界条件の制約を守りつつ偏微分方程式の解を求めるディープラーニングモデル。Amazon ScienceからICMLとICLRで発表</li>
</ul>
</li>
<li>LLMでドメイン特化言語を作りまくり？
<ul>
<li><a href="https://huggingface.co/papers/2305.19234">https://huggingface.co/papers/2305.19234</a></li>
<li>Grammar Prompting for Domain-Specific Language Generation with Large Language Models</li>
</ul>
</li>
<li>OpenAI のCEOのインタビュー、GPUリソースが世界的に足らないのはGPT-4のマルチモーダル学習中のため？
<ul>
<li><a href="https://humanloop.com/blog/openai-plans">https://humanloop.com/blog/openai-plans</a></li>
</ul>
</li>
<li>「AIに脅かされる「個人」　情報を断ち切る規制必要」　政治季評
<ul>
<li><a href="https://www.asahi.com/articles/ASR5065XLR5YUSPT006.html">https://www.asahi.com/articles/ASR5065XLR5YUSPT006.html</a></li>
<li>中身は議論されず、最初のChatGPTはオープンソースのところに、鬼首をとったようにかみつくDS界隈</li>
</ul>
</li>
<li>QCDの１方程式から多様な世界が作り出されるチャート
<ul>
<li><a href="http://suganuma-hideo.o.oo7.jp/hideo/index.files/main.files/HQCD.pdf">http://suganuma-hideo.o.oo7.jp/hideo/index.files/main.files/HQCD.pdf</a></li>
</ul>
</li>
<li>“According to . . . ” Prompting Language Models Improves Quoting from Pre-Training Data
<ul>
<li>Wikipediaによると、、、を付け加えるプロンプトテクニック？</li>
<li>LLMが事前学習データから直接引用するように誘導し、生成される情報の信頼性を向上</li>
<li><a href="https://arxiv.org/pdf/2305.13252.pdf">https://arxiv.org/pdf/2305.13252.pdf</a></li>
</ul>
</li>
<li>inna-3.6b-instruction-oppのggml 4q_2を作って、LangChainのsummarize chainで使ってみました…
<ul>
<li><a href="https://twitter.com/8hmVmEGJ6nFyUE5/status/1663936372363898880?s=20">https://twitter.com/8hmVmEGJ6nFyUE5/status/1663936372363898880?s=20</a></li>
<li>やっぱりggmlと4bitが最強なのか。。モデルサイズが2Gって、あーた</li>
</ul>
</li>
<li>LLMをつかって、微分方程式から保存則を抽出する？？
<ul>
<li>Discovering New Interpretable Conservation Laws as Sparse Invariants</li>
<li><a href="https://arxiv.org/abs/2305.19525">https://arxiv.org/abs/2305.19525</a></li>
</ul>
</li>
<li>OpenAIがsecurity portalを公開
<ul>
<li><a href="https://trust.openai.com/">https://trust.openai.com/</a></li>
</ul>
</li>
<li>つい最近引退された、ストラング教授（線形代数他）のインタビュー記事
<ul>
<li><a href="https://news.mit.edu/2023/gilbert-strang-made-linear-algebra-fun-0531">https://news.mit.edu/2023/gilbert-strang-made-linear-algebra-fun-0531</a></li>
</ul>
</li>
<li>GPT4ALLをつかって、GPUなしで、ローカルPCでLLMを動かす
<ul>
<li><a href="https://gpt4all.io/index.html">https://gpt4all.io/index.html</a></li>
<li>A free-to-use, locally running, privacy-aware chatbot. <strong>No GPU or internet required.</strong></li>
</ul>
</li>
<li>ChatGPTプラグイン「Notable」だけでデータ分析コンペに挑戦してみた話
<ul>
<li><a href="https://qiita.com/ot12/items/ba74fa150e160d94a71f">https://qiita.com/ot12/items/ba74fa150e160d94a71f</a></li>
<li>やっぱりNoteableは最強の件、来年のデータサイエンス特論のネタにしよう！</li>
</ul>
</li>
<li>A Mechanistic Interpretability Analysis of Grokking
<ul>
<li>学習が進むと突然、未見のデータに一般化するように学習する現象のメカニズムの解明だそうだ</li>
<li><a href="https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking">https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking</a></li>
</ul>
</li>
<li>カモシカ-LoRaから、OpenCALM 7B, 3Bをファインチューニングして作成したアダプタを公開
<ul>
<li><a href="https://twitter.com/kam0shika/status/1663906516276051969?s=20">https://twitter.com/kam0shika/status/1663906516276051969?s=20</a></li>
</ul>
</li>
<li>Transformer.js、ブラウザやnodejsからhuggingfaceのtransformerが使える
<ul>
<li><a href="https://github.com/xenova/transformers.js">https://github.com/xenova/transformers.js</a></li>
</ul>
</li>
<li>特異値分解で30%もLLMを圧縮しても性能が変わらなかった
<ul>
<li>LLMってやっぱり疎なのね</li>
<li>~30% Compression Of LLM (Flan-T5-Base) With Low Rank Decomposition Of Attention Weight Matrices</li>
<li><a href="https://smashinggradient.com/2023/05/23/30-compression-of-llms-with-low-rank-decomposition-of-attention-weight-matrices/">https://smashinggradient.com/2023/05/23/30-compression-of-llms-with-low-rank-decomposition-of-attention-weight-matrices/</a></li>
</ul>
</li>
<li>LQML;
<ul>
<li>LMQL (Language Model Query Language) is a programming language for large language model (LM) interaction.</li>
<li><a href="https://docs.lmql.ai/en/stable/">https://docs.lmql.ai/en/stable/</a></li>
</ul>
</li>
<li>Andrew Ngさんによる米軍AIドローンシミュレーション（操作者を殺すという結論）への反駁
<ul>
<li><a href="https://twitter.com/AndrewYNg/status/1664694504476102680?s=20">https://twitter.com/AndrewYNg/status/1664694504476102680?s=20</a></li>
</ul>
</li>
<li>スタンフォード大学による、機械学習もろもろチートシート
<ul>
<li><a href="https://github.com/afshinea/stanford-cs-229-machine-learning">https://github.com/afshinea/stanford-cs-229-machine-learning</a></li>
</ul>
</li>
<li>LangChainをサポートするvicuna-13bモデルをrinnaが公開
<ul>
<li><a href="https://huggingface.co/rinna/vicuna-13b-delta-finetuned-langchain-MRKL">https://huggingface.co/rinna/vicuna-13b-delta-finetuned-langchain-MRKL</a></li>
<li><a href="https://note.com/hamachi_jp/n/n97d368a617ac">https://note.com/hamachi_jp/n/n97d368a617ac</a></li>
</ul>
</li>
<li>A Survey on Large Language Models for Recommendation
<ul>
<li><a href="https://arxiv.org/abs/2305.19860v2">https://arxiv.org/abs/2305.19860v2</a></li>
</ul>
</li>
<li>分子生物学にLLMが最適な件
<ul>
<li><a href="https://towardsdatascience.com/large-language-models-in-molecular-biology-9eb6b65d8a30">https://towardsdatascience.com/large-language-models-in-molecular-biology-9eb6b65d8a30</a></li>
</ul>
</li>
<li>GPT4ALLとLangChainとChromaをつかった、ローカルに動く最小限のQ&amp;A
<ul>
<li><a href="https://twitter.com/AssemblyAI/status/1661747770108305409?s=20">https://twitter.com/AssemblyAI/status/1661747770108305409?s=20</a></li>
</ul>
</li>
<li>「ChatGPTの出現は自然言語処理の専門家に何を問いかけているか」
<ul>
<li>言語学会の乾先生の巻頭言</li>
<li>「では，これで自然言語処理は終わるのか？ もちろん，終わらない．解くべき課題，新たに生まれる問いは山ほどある．」</li>
<li><a href="https://www.anlp.jp/topics/topic230601.html">https://www.anlp.jp/topics/topic230601.html</a></li>
</ul>
</li>
<li>「言語の本質　ことばはどう生まれ、進化したか (中公新書)」
<ul>
<li>今井むつみ, 秋田喜美の本、</li>
<li><a href="https://www.amazon.co.jp/dp/B0C4XF523T?ref_=k4w_ss_dp_lp">https://www.amazon.co.jp/dp/B0C4XF523T?ref_=k4w_ss_dp_lp</a></li>
</ul>
</li>
<li>Langchain・Semantic Kernel・guidanceでエージェント機能を実装して比較してみた
<ul>
<li><a href="https://qiita.com/sakue_103/items/6ffee0bc267e71eafd60">https://qiita.com/sakue_103/items/6ffee0bc267e71eafd60</a></li>
</ul>
</li>
<li>時系列データにおける特徴量エンジニアリング by NRI
<ul>
<li><a href="https://datascience.nri.com/entry/2022/10/12/155350">https://datascience.nri.com/entry/2022/10/12/155350</a></li>
</ul>
</li>
<li>ドイツ連邦データ保護当局（BfDI）の生成AIについての声明（5月22日）、 by 生貝先生
<ul>
<li><a href="https://www.bfdi.bund.de/SharedDocs/Downloads/DE/DokumenteBfDI/Stellungnahmen/2023/StgN_Generative-K%C3%BCnstliche-Intelligenz.pdf?__blob=publicationFile&amp;v=2">https://www.bfdi.bund.de/SharedDocs/Downloads/DE/DokumenteBfDI/Stellungnahmen/2023/StgN_Generative-Künstliche-Intelligenz.pdf?__blob=publicationFile&amp;v=2</a></li>
<li>GDPR的なリスクベースアプローチとDSA的なシステミックリスクアプローチの対比など興味深い。青少年学習データのフィルタリングやAI規則の川上川下問題なども</li>
</ul>
</li>
<li>Jupyter AIが出た！試した！！すごい！！！
<ul>
<li><a href="https://qiita.com/moritalous/items/a270d5932ebee18d0ba8?utm_content=buffer352b5&amp;utm_medium=social&amp;utm_source=twitter.com&amp;utm_campaign=buffer">https://qiita.com/moritalous/items/a270d5932ebee18d0ba8?utm_content=buffer352b5&amp;utm_medium=social&amp;utm_source=twitter.com&amp;utm_campaign=buffer</a></li>
<li>すごいらしい</li>
</ul>
</li>
<li></li>
</ul>
<h2 id="section-16">5/29</h2>
<p>Microsoft BuildでWindowsとGPTとの統合とか、BingでもChatGPTのプラグインが使えるようになるとか、相も変わらずMicrosoftはどうやって投資を回収できるのか不明。Adamを超えるSophiaの登場や、4bit化のQLoRaの登場など、個人や企業でのLLM作成には朗報であるが、LLMの民主化って危険もあるよね。なので、DeepMindやMicrosoftは倫理性やリスクに関する研究をちゃんと続けて公開している。Voyagerすごい、研究開発のタスクももGPT-4でできるのでは？WebGPUを使ったwebllm、nodejs版でも動く模様。microsoftはguidanceでモデル利用の効率化の工夫を行っているとの報告も。アブダビから謎の巨大LLMであるFalcon-40Bが発表されるも、OSSとうたいつつ実は謎ライセンスであっというまに叩かれる。SQLとかNotableとかデータサイエンス系のChatGPTの活用が本格的に。祝！SIAMの賞の受賞、蔵本先生！！！。欧州AI規制の最終投票を目前に、OpenAI、欧州AI規制遵守が困難と判断されれば、欧州からサービス引き上げとの記事。ChatGPTアプリが日本でもiPhoneに登場、似たような名前のアプリがたくさんあって、、、。</p>
<ul>
<li>MicorsoftのAIが倫理的であるかどうかを評価するツールキット
<ul>
<li><a href="https://github.com/microsoft/responsible-ai-toolbox">https://github.com/microsoft/responsible-ai-toolbox</a></li>
</ul>
</li>
<li>知識グラフのneo4jと、LangChainからの利用、Cypher問い合わせを自動生成する
<ul>
<li><a href="https://python.langchain.com/en/latest/modules/chains/examples/graph_cypher_qa.html">https://python.langchain.com/en/latest/modules/chains/examples/graph_cypher_qa.html</a></li>
</ul>
</li>
<li>LIMA: Less Is More for Alignment
<ul>
<li>Lucan先生によると、LLaMA 65B + 1000 supervised samples = {GPT4, Bard} level performance</li>
<li><a href="https://arxiv.org/abs/2305.11206">https://arxiv.org/abs/2305.11206</a></li>
</ul>
</li>
<li>scikit-llm: scikit-learnとLLMをシームレスつにつなげる
<ul>
<li><a href="https://github.com/iryna-kondr/scikit-llm">https://github.com/iryna-kondr/scikit-llm</a></li>
</ul>
</li>
<li>LangChainからAzure OpenAI を使うメモ
<ul>
<li><a href="https://qiita.com/tmiyata25/items/7a04096342241d8a2b4c">https://qiita.com/tmiyata25/items/7a04096342241d8a2b4c</a></li>
</ul>
</li>
<li>Textually Pretrained Speech Language Models：なんか音声をいれると音声を出力するLLM!!
<ul>
<li><a href="https://pages.cs.huji.ac.il/adiyoss-lab/twist/">https://pages.cs.huji.ac.il/adiyoss-lab/twist/</a></li>
</ul>
</li>
<li>ImageBind　by Meta、マルチモーダルな学習
<ul>
<li><a href="https://ai.facebook.com/blog/imagebind-six-modalities-binding-ai/?utm_source=twitter&amp;utm_medium=organic_social&amp;utm_campaign=blog&amp;utm_content=card">https://ai.facebook.com/blog/imagebind-six-modalities-binding-ai/?utm_source=twitter&amp;utm_medium=organic_social&amp;utm_campaign=blog&amp;utm_content=card</a></li>
</ul>
</li>
<li>open-calm-7b を databricks-dolly-15k-ja で LoRA したのをマージして ggml にして 4bit 量子化して redpajama.cpp で MacBook ローカルで動く日本語高速チャットボット
<ul>
<li><a href="https://twitter.com/niw/status/1660894493867134976?s=20">https://twitter.com/niw/status/1660894493867134976?s=20</a></li>
</ul>
</li>
<li>LLaMAベースの日本語大規模言語モデル(LoRaした)公開
<ul>
<li><a href="https://llm.msuzuki.me/">https://llm.msuzuki.me/</a></li>
</ul>
</li>
<li>Microsoft Build開催、OSとLLMが融合？
<ul>
<li><a href="https://blogs.microsoft.com/blog/2023/05/23/microsoft-build-brings-ai-tools-to-the-forefront-for-developers/">https://blogs.microsoft.com/blog/2023/05/23/microsoft-build-brings-ai-tools-to-the-forefront-for-developers/</a></li>
</ul>
</li>
<li>LLM向けの学習最適化エンジンSophia、アダムを超えるか
<ul>
<li>Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training</li>
<li><a href="https://arxiv.org/abs/2305.14342">https://arxiv.org/abs/2305.14342</a></li>
</ul>
</li>
<li>QLoRa: HuggingFaceのモデルが、4bit化されたものが使えるようになる？
<ul>
<li><a href="https://huggingface.co/blog/4bit-transformers-bitsandbytes">https://huggingface.co/blog/4bit-transformers-bitsandbytes</a></li>
</ul>
</li>
<li>Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks
<ul>
<li><a href="https://huggingface.co/papers/2305.14201">https://huggingface.co/papers/2305.14201</a></li>
</ul>
</li>
<li>自然言語のpromptによるLLMの利用は、LLMの本来の能力を生かしきれてない
<ul>
<li><a href="https://arxiv.org/abs/2305.13264v1">https://arxiv.org/abs/2305.13264v1</a></li>
</ul>
</li>
<li>QLoRaを使えば、普通のcolabにて、数時間でLLMができるという報告。4bit最強。
<ul>
<li>33B-parameter LLM on Google Colab in a few hour</li>
<li><a href="https://twitter.com/ItakGol/status/1661714548594823174?s=20">https://twitter.com/ItakGol/status/1661714548594823174?s=20</a></li>
</ul>
</li>
<li>OpenCALM-7BをLoRAでFine tuningして対話ができるようにする
<ul>
<li><a href="https://note.com/masuidrive/n/n0e2a11fc5bfa">https://note.com/masuidrive/n/n0e2a11fc5bfa</a></li>
</ul>
</li>
<li>Reasoning with Language Model is Planning with World Model
<ul>
<li>CoT on GPT-4との比較で勝るとのこと</li>
<li><a href="https://arxiv.org/abs/2305.14992">https://arxiv.org/abs/2305.14992</a></li>
</ul>
</li>
<li>Voyager: 長期的な探索をGPT-4でやらせる例。Minecraftをやらせたら、、（研究開発も。。。）
<ul>
<li><a href="https://github.com/MineDojo/Voyager">https://github.com/MineDojo/Voyager</a></li>
</ul>
</li>
<li>祝！蔵本先生、SIAMでJürgen Moser Lecture賞受賞！受賞講演
<ul>
<li><a href="https://www.youtube.com/watch?v=2P-EgTSa-E4&amp;feature=youtu.be">https://www.youtube.com/watch?v=2P-EgTSa-E4&amp;feature=youtu.be</a></li>
</ul>
</li>
<li>DeepMindから、一般的なAIモデルが潜在的に持ちうる有害なリスクを評価するフレームワーク
<ul>
<li><a href="https://www.deepmind.com/blog/an-early-warning-system-for-novel-ai-risks?utm_source=twitter&amp;utm_medium=social&amp;utm_campaign=ModelEval">https://www.deepmind.com/blog/an-early-warning-system-for-novel-ai-risks?utm_source=twitter&amp;utm_medium=social&amp;utm_campaign=ModelEval</a></li>
</ul>
</li>
<li>WebGPUをつかってLLMを動かす仕組み、WebLLMが、nodejsでも動く？？
<ul>
<li><a href="https://github.com/mlc-ai/web-llm">https://github.com/mlc-ai/web-llm</a></li>
</ul>
</li>
<li>LangChainからDatabricksを使う
<ul>
<li><a href="https://python.langchain.com/en/latest/modules/models/llms/integrations/databricks.html">https://python.langchain.com/en/latest/modules/models/llms/integrations/databricks.html</a></li>
</ul>
</li>
<li>アブダビの研究所からFalcon-40Bが発表、オープンソースなのに、ライセンス料が必要みたいな罠が見つかる。
<ul>
<li><a href="https://huggingface.co/tiiuae/falcon-40b">https://huggingface.co/tiiuae/falcon-40b</a></li>
<li>「売上の10%をロイヤリティとして12ヶ月毎に支払わなければならない」</li>
</ul>
</li>
<li>microsoft/guidance(LangChainのようなもの）をつかって、Agentを定義して、動かす
<ul>
<li><a href="https://note.com/explaza_inc/n/n7cb8043506bd">https://note.com/explaza_inc/n/n7cb8043506bd</a></li>
</ul>
</li>
<li>OpenAIがgptサービスの５月の速度低下をレポートするものが
<ul>
<li><a href="https://twitter.com/helicone_ai/status/1662325356563496961?s=20">https://twitter.com/helicone_ai/status/1662325356563496961?s=20</a></li>
</ul>
</li>
<li>Googleが生成AIの無料講座を公開
<ul>
<li><a href="https://www.cloudskillsboost.google/journeys/118">https://www.cloudskillsboost.google/journeys/118</a></li>
</ul>
</li>
<li>SQLを活用したデータ分析におけるChatGPTの活用法
<ul>
<li><a href="https://speakerdeck.com/hikarut/sqlwohuo-yong-sitadetafen-xi-niokeruchatgptnohuo-yong-fa">https://speakerdeck.com/hikarut/sqlwohuo-yong-sitadetafen-xi-niokeruchatgptnohuo-yong-fa</a></li>
</ul>
</li>
<li>ChatGPTのデータサイエンス向けのプラグインNotableが便利との記事
<ul>
<li><a href="https://secon.dev/entry/2023/05/27/170000-noteable-iris/">https://secon.dev/entry/2023/05/27/170000-noteable-iris/</a></li>
</ul>
</li>
<li>Lucan先生、GAFAMの代わりに、MAGMAを造語。 Meta, Amazon, Google, Microsoft, App
<ul>
<li><a href="https://twitter.com/ylecun/status/1662375684612685825?s=20">https://twitter.com/ylecun/status/1662375684612685825?s=20</a></li>
</ul>
</li>
<li>OpenAIのアルトマンCEO、「EU AI Act順守が困難ならEUでの事業は停止する
<ul>
<li><a href="https://www.itmedia.co.jp/news/articles/2305/26/news106.html">https://www.itmedia.co.jp/news/articles/2305/26/news106.html</a></li>
</ul>
</li>
</ul>
<h2 id="section-17">5/22</h2>
<p>ChatGPT以外のOSSのLLMでは、googleのFLAN-20B with UL2 ぐらいならば、なんとか同等の性能がでるという報告も(A100が必要)。privateGPTや、GPT4ALLなどの、ローカル環境で動かせるOSSのLLMもだいぶそろってきました。PyTorchをブラウザ環境’(TypeScriptで）動かす仕組みも登場。しかし本命は、WebGPUをつかって、ブラウザ以外からもLLMをローカルで高速に動かす試みには期待したいところ。いっぽうTinyStoriesなど、どれだけLLMを小さくできるかな？的なアプローチも続く。Tramnsformerも偏微方程式を解くなど、物理モデルの領域に広げる試みも。日本からは日本語版LLMが複数出現、実力のほどは？？LLMの説明性やバイアス対策なども。ChatGPTがついにIPhoneに乗る（USのみ）。MicrosoftはLLMを使いやすくするフレームワークGuidanceを発表、SemanticKernelの立場は？？Marvinのような、LLMとプログラミングの融合パラダイムには可能性がありそうです。</p>
<ul>
<li>LLMのバイアスをあぶりだす、Constructive Input Decoding(CID) by google
<ul>
<li><a href="https://arxiv.org/abs/2305.07378">https://arxiv.org/abs/2305.07378</a></li>
</ul>
</li>
<li>privateGPT:ローカル環境で動く最小限のGPT、LangChain, GPT4All, LlamaCpp, Chroma and SentenceTransformersを活用
<ul>
<li><a href="https://github.com/imartinez/privateGPT">https://github.com/imartinez/privateGPT</a></li>
</ul>
</li>
<li>TinyStories:３～４才ぐらいが理解できる短い文書のデータセット、どれだけLLMを小さくできるかを評価するためのもの by Microsoft
<ul>
<li><a href="https://arxiv.org/abs/2305.07759">https://arxiv.org/abs/2305.07759</a></li>
</ul>
</li>
<li>Google/OpenAIがオープンソースのLLMを開発している。
<ul>
<li><a href="https://www.theinformation.com/articles/open-source-ai-is-gaining-on-google-and-chatgpt">https://www.theinformation.com/articles/open-source-ai-is-gaining-on-google-and-chatgpt</a></li>
</ul>
</li>
<li>Marvin:プログラミングとLLMの補助を組み合わせた新しいパラダイム、LMQLみたいな感じ？スキーマに従ってデータ抽出など
<ul>
<li><a href="https://note.com/hamachi_jp/n/na1960fc9d6d3">https://note.com/hamachi_jp/n/na1960fc9d6d3</a></li>
<li><a href="https://www.askmarvin.ai/">https://www.askmarvin.ai/</a></li>
</ul>
</li>
<li>Excelとチャットする、titnanicの例で、前処理のところをチャットで実現
<ul>
<li><a href="https://github.com/Anil-matcha/Chat-With-Excel/blob/main/Data_analysis_with_langchain.ipynb">https://github.com/Anil-matcha/Chat-With-Excel/blob/main/Data_analysis_with_langchain.ipynb</a></li>
</ul>
</li>
<li>Physics Informed Token Transformer(PITT)：偏微分方程式(PDE)をトークン化してエンベディングし、PDEの解を求める機械学習手法として有名なFourier Neural Operator(FNO)の補正として利用
<ul>
<li><a href="https://arxiv.org/abs/2305.08757v1">https://arxiv.org/abs/2305.08757v1</a></li>
</ul>
</li>
<li>Abbeel教授によるHinton教授へのインタビュー、NYTimesの記事依頼、全世界から２分毎ｎ取材依頼が来たらしい
<ul>
<li><a href="https://www.youtube.com/watch?v=rLG68k2blOc">https://www.youtube.com/watch?v=rLG68k2blOc</a></li>
</ul>
</li>
<li>医療分野に特化した言語モデル「Med-PaLM2」の論文、現役の医者もPaLM2の回答のほうを評価
<ul>
<li><a href="https://arxiv.org/abs/2305.09617">https://arxiv.org/abs/2305.09617</a></li>
</ul>
</li>
<li>rinna、日本語に特化した36億パラメータのGPT言語モデルを公開
<ul>
<li><a href="https://rinna.co.jp/news/2023/05/20230507.html">https://rinna.co.jp/news/2023/05/20230507.html</a></li>
</ul>
</li>
<li>MicrosoftがLangchainみたいな、Guidanceを発表
<ul>
<li><a href="https://github.com/microsoft/guidance">https://github.com/microsoft/guidance</a></li>
</ul>
</li>
<li>CyberAgentが日本語版ローカルLLMを発表
<ul>
<li><a href="https://huggingface.co/cyberagent">https://huggingface.co/cyberagent</a></li>
</ul>
</li>
<li>Google の FLAN-20B with UL2 レベルならば、ChatGPT APIのように使えるらしい
<ul>
<li><a href="https://qiita.com/sakasegawa/items/7394fe68eb0087b3c4a5">https://qiita.com/sakasegawa/items/7394fe68eb0087b3c4a5</a></li>
</ul>
</li>
<li>Google、自社のcolabratoryに、コード生成機能を搭載するらしい
<ul>
<li><a href="https://blog.google/technology/developers/google-colab-ai-coding-features/">https://blog.google/technology/developers/google-colab-ai-coding-features/</a></li>
</ul>
</li>
<li>Transformer.js: Hugging Faceのtransformerを、ブラウザで動かすことができる、ONIX runtimeを利用、WebGPU対応は不明
<ul>
<li><a href="https://github.com/xenova/transformers.js">https://github.com/xenova/transformers.js</a></li>
</ul>
</li>
<li>Graph Neural Network(GNN)で、巡回セールスマン問題を解く、スタンフォード大学の講義での事例、CS224W
<ul>
<li><a href="https://medium.com/stanford-cs224w/tackling-the-traveling-salesman-problem-with-graph-neural-networks-b86ef4300c6e">https://medium.com/stanford-cs224w/tackling-the-traveling-salesman-problem-with-graph-neural-networks-b86ef4300c6e</a></li>
</ul>
</li>
<li>OpenCALM-7Bをdolly-15k-jaでLoRAしたら、ある程度会話できるようになった
<ul>
<li><a href="https://twitter.com/masuidrive/status/1659089478781227008?s=20">https://twitter.com/masuidrive/status/1659089478781227008?s=20</a></li>
</ul>
</li>
<li>LLMの出力の説明に関する論文らしい、Explaining black box text modules in natural language with language models by microsoft
<ul>
<li><a href="https://huggingface.co/papers/2305.09863">https://huggingface.co/papers/2305.09863</a></li>
</ul>
</li>
<li>TokenHawk、WebGPUを活用して、ローカルで、WebでLLMを動かすことができる仕組み、GoogleのDawnエンジン利用
<ul>
<li><a href="https://github.com/kayvr/token-hawk">https://github.com/kayvr/token-hawk</a></li>
</ul>
</li>
<li>ChatGPTがiPhoneで動くようになる(米国)
<ul>
<li><a href="https://openai.com/blog/introducing-the-chatgpt-app-for-ios">https://openai.com/blog/introducing-the-chatgpt-app-for-ios</a></li>
</ul>
</li>
<li>Trasnformerを制御に用いる、# A Generalist Dynamics Model for Control、by DeepMind
<ul>
<li><a href="https://huggingface.co/papers/2305.10912">https://huggingface.co/papers/2305.10912</a></li>
</ul>
</li>
<li>LangChainから、Spark SQL Agent
<ul>
<li><a href="https://python.langchain.com/en/latest/modules/agents/toolkits/examples/spark_sql.html">https://python.langchain.com/en/latest/modules/agents/toolkits/examples/spark_sql.html</a></li>
</ul>
</li>
<li>LangChainから、ローカルにダウンロードしたGPT4ALLの使い方改善
<ul>
<li><a href="https://python.langchain.com/en/latest/modules/models/llms/integrations/gpt4all.html">https://python.langchain.com/en/latest/modules/models/llms/integrations/gpt4all.html</a></li>
</ul>
</li>
<li>Language Models Meet World Models: Embodied Experiences Enhance Language Models
<ul>
<li><a href="https://arxiv.org/abs/2305.10626">https://arxiv.org/abs/2305.10626</a></li>
</ul>
</li>
<li>WebGPU-pytorch、pytorchが、webGPUの上で動く（学習、推論とも）
<ul>
<li><a href="https://github.com/praeclarum/webgpu-torch">https://github.com/praeclarum/webgpu-torch</a></li>
</ul>
</li>
<li>Hugging FaceのモデルをLangChainで使う方法を調べた、Hubを使うか、ローカルにダウンロードして使うか、
<ul>
<li><a href="https://www.mattari-benkyo-note.com/2023/05/19/langchain_hugging_face/">https://www.mattari-benkyo-note.com/2023/05/19/langchain_hugging_face/</a></li>
</ul>
</li>
<li>Stanford大学のTransformerの事業CS２５が最強の件
<ul>
<li><a href="https://web.stanford.edu/class/cs25/">https://web.stanford.edu/class/cs25/</a></li>
</ul>
</li>
<li>Stanford大学、文字列のアライメントライブラリstring2string
<ul>
<li><a href="https://github.com/stanfordnlp/string2string">https://github.com/stanfordnlp/string2string</a></li>
</ul>
</li>
<li>Self-Queringという手法、による文書検索Weaviate、スキーマを与えると、検索結果に、情報抽出の結果も出してくれる（曲のratingとかgeneとかのメタデータなど）もやってくれる。おおすごい
<ul>
<li><a href="https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/weaviate_self_query.html">https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/weaviate_self_query.html</a></li>
</ul>
</li>
<li>BCGがまとめた日本企業の変革を阻む「チェンジモンスター」資料、ポケモン的なキャラクター付け
<ul>
<li><a href="https://web-assets.bcg.com/img-src/japan%20tembo-146-change%20monster_1oct2002_tcm9-169992.pdf">https://web-assets.bcg.com/img-src/japan tembo-146-change monster_1oct2002_tcm9-169992.pdf</a></li>
</ul>
</li>
</ul>
<h2 id="section-18">5/15</h2>
<ul>
<li>オリジナルのTransformer論文のアーキテクチャ構成の絵が、本文と合ってないと記事が、
<ul>
<li><a href="https://arxiv.org/abs/2002.04745">https://arxiv.org/abs/2002.04745</a></li>
</ul>
</li>
<li>few-shot learningで満足できない人の応用プロンプト集
<ul>
<li><a href="https://cameronrwolfe.substack.com/p/advanced-prompt-engineering">https://cameronrwolfe.substack.com/p/advanced-prompt-engineering</a></li>
</ul>
</li>
<li>アッセンブリ理論、有機化合物で分子の結合の複雑さの評価、LLMの評価にも使える？
<ul>
<li><a href="https://www.quantamagazine.org/a-new-theory-for-the-assembly-of-life-in-the-universe-20230504/">https://www.quantamagazine.org/a-new-theory-for-the-assembly-of-life-in-the-universe-20230504/</a></li>
</ul>
</li>
<li>AGIが人類を壊滅させる可能性はほぼ100%といった強い悲観論、AI Alignment Centerの人の話によると、
<ul>
<li><a href="https://note.com/bioshok/n/n43041a52a529">https://note.com/bioshok/n/n43041a52a529</a></li>
</ul>
</li>
<li>OpenAIが公開した、プロンプトから3Dモデルを作るShap-Eのデモサイトがhuggingfaceに。
<ul>
<li><a href="https://huggingface.co/spaces/hysts/Shap-E">https://huggingface.co/spaces/hysts/Shap-E</a></li>
</ul>
</li>
<li>LLamaindexに新しいドキュメント要約の仕組みが導入？
<ul>
<li><a href="https://medium.com/llamaindex-blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec">https://medium.com/llamaindex-blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec</a></li>
</ul>
</li>
<li>OpenAI: GPT-4で、GPT-2の個々のニューロンの働きの説明というか、意味づけを行う？？
<ul>
<li><a href="https://openai.com/research/language-models-can-explain-neurons-in-language-models">https://openai.com/research/language-models-can-explain-neurons-in-language-models</a></li>
</ul>
</li>
<li>VAEによる分子生成のモデルの改良の話
<ul>
<li><a href="https://arxiv.org/abs/2305.03041v1">https://arxiv.org/abs/2305.03041v1</a></li>
</ul>
</li>
<li>GTP-4日本の医師国家試験で合格？
<ul>
<li><a href="https://news.yahoo.co.jp/articles/60da4c733c2a03a9829bc598f8dcc246e4d10b00">https://news.yahoo.co.jp/articles/60da4c733c2a03a9829bc598f8dcc246e4d10b00</a></li>
</ul>
</li>
<li>LLamaindexにて、正式にhuggingfaceの LLM support　される
<ul>
<li><a href="https://github.com/jerryjliu/llama_index">https://github.com/jerryjliu/llama_index</a></li>
</ul>
</li>
<li>WebGPUで、LLMをローカルに動かす動きが活発に、LaMA, Alpaca, Vicuna, and Dol
<ul>
<li><a href="https://github.com/mlc-ai/web-llm">https://github.com/mlc-ai/web-llm</a></li>
</ul>
</li>
<li>Google I/OでPaLM 2を発表
<ul>
<li><a href="https://ai.google/static/documents/palm2techreport.pdf">https://ai.google/static/documents/palm2techreport.pdf</a></li>
</ul>
</li>
<li>Wikipediaに対するQ&amp;Aを可能にするretreaverを提供するCoheare?
<ul>
<li><a href="https://github.com/menloparklab/cohere-weaviate-wikipedia-retrieval">https://github.com/menloparklab/cohere-weaviate-wikipedia-retrieval</a></li>
<li><a href="https://github.com/weaviate/weaviate">https://github.com/weaviate/weaviate</a></li>
</ul>
</li>
<li>Google I/Oが大収穫だった模様、Bardは日本語、韓国語対応、 Bard、PaLM 2
<ul>
<li><a href="https://www.gizmodo.jp/2023/05/google-io23-ai-outline.html">https://www.gizmodo.jp/2023/05/google-io23-ai-outline.html</a></li>
</ul>
</li>
<li>BardとGPT-4の性能比較、結構GPT-4に肉薄している模様。
<ul>
<li><a href="https://qiita.com/kumag0r0/items/77dbe743643183ae3e98">https://qiita.com/kumag0r0/items/77dbe743643183ae3e98</a></li>
</ul>
</li>
<li>Bard発表のプレゼンで、「日本語」のフォントが残念と話題に、、、
<ul>
<li><a href="https://www.itmedia.co.jp/news/articles/2305/11/news178.html">https://www.itmedia.co.jp/news/articles/2305/11/news178.html</a></li>
</ul>
</li>
<li>HumanML3D:Human motion language Dataset
<ul>
<li><a href="https://github.com/EricGuo5513/HumanML3D">https://github.com/EricGuo5513/HumanML3D</a></li>
</ul>
</li>
<li>DeepL日本に拠点を置く？
<ul>
<li><a href="https://newsdig.tbs.co.jp/articles/-/480597?display=1">https://newsdig.tbs.co.jp/articles/-/480597?display=1</a></li>
</ul>
</li>
<li>GoogleのPhotorealistic 3D Tilesを<a href="https://t.co/j5x1oduUK1">http://deck.gl</a>で表示、軽いらしい
<ul>
<li><a href="https://twitter.com/syanseto/status/1656586481094520838?s=20">https://twitter.com/syanseto/status/1656586481094520838?s=20</a></li>
<li>deck.lg(TerrainExtension) + Google Photorealistic 3D Tiles</li>
<li>Google 3D tileで読み込んだ3Dモデルの上にTerrainExtensionを使ってGeoJSONポリゴンをオーバーレイ</li>
</ul>
</li>
<li>日本語T5モデルの公開 by レトリバ
<ul>
<li><a href="https://note.com/retrieva/n/n7b4186dc5ada">https://note.com/retrieva/n/n7b4186dc5ada</a></li>
</ul>
</li>
<li>LeCun先生の講演、LeCun: Towards Machines That Can Understand, Reason, &amp; Plan
<ul>
<li><a href="https://www.youtube.com/watch?v=_JfEScYyVCE">https://www.youtube.com/watch?v=_JfEScYyVCE</a></li>
</ul>
</li>
<li>OpenAI、ChatGPT Plusユーザー全体に、5/12よりPluginが使えるようなるとアナウンス
<ul>
<li><a href="https://help.openai.com/en/articles/6825453-chatgpt-release-notes">https://help.openai.com/en/articles/6825453-chatgpt-release-notes</a></li>
</ul>
</li>
<li>拡散モデルを用いることで２次元の分子グラフからでも同等の励起状態の予測精度が得られるという話らしい
<ul>
<li><a href="https://arxiv.org/abs/2304.12233v2">https://arxiv.org/abs/2304.12233v2</a></li>
</ul>
</li>
<li>機械学習理論発展、Hyperbolic Poincaré distributions = Probability distributions with support the Poincaré disk
<ul>
<li><a href="https://arxiv.org/abs/2205.13984">https://arxiv.org/abs/2205.13984</a></li>
</ul>
</li>
<li>Graph Transformer (GT)を作る例題
<ul>
<li><a href="https://arxiv.org/pdf/2012.09699.pdf">https://arxiv.org/pdf/2012.09699.pdf</a></li>
</ul>
</li>
<li>推薦において、ユーザーの嗜好って、LLMは本当に理解してるんだったけ論文。
<ul>
<li><a href="https://arxiv.org/abs/2305.06474">https://arxiv.org/abs/2305.06474</a></li>
</ul>
</li>
<li>量子機械学習の研究者が、軒並み量子をやめて機械学習にいってるという、組合せ最適化の大家である湊先生の嘆き
<ul>
<li><a href="https://twitter.com/MinatoYuichiro/status/1657243184064499712?s=20">https://twitter.com/MinatoYuichiro/status/1657243184064499712?s=20</a></li>
</ul>
</li>
<li>GoogleのPhotorealistic 3D Tiles（左）と国交省の3D都市モデルPLATEAUの3D Tiles（右）の比較
<ul>
<li><a href="https://twitter.com/syanseto/status/1656964913913540608?s=20">https://twitter.com/syanseto/status/1656964913913540608?s=20</a></li>
</ul>
</li>
<li>ChatGPTとOSSのLLM達とガチタスクでの比較、いい線言ってるらしい。Vicuna-13B, ChatGPT (3.5), MPT-7B-Chat
<ul>
<li><a href="https://medium.com/@marcotcr/exploring-chatgpt-vs-open-source-models-on-slightly-harder-tasks-aa0395c31610">https://medium.com/@marcotcr/exploring-chatgpt-vs-open-source-models-on-slightly-harder-tasks-aa0395c31610</a></li>
</ul>
</li>
<li>PrivateGPT:単にOSSのLLMをダウンロードしてチャットに仕立てる、LangChain and GPT4All and LlamaCpp
<ul>
<li><a href="https://github.com/imartinez/privateGPT">https://github.com/imartinez/privateGPT</a></li>
</ul>
</li>
<li>OpenAIの Sam Altman氏の、謎のツイート"summer is coming"
<ul>
<li><a href="https://twitter.com/sama/status/1657405294354518017?s=20">https://twitter.com/sama/status/1657405294354518017?s=20</a></li>
</ul>
</li>
<li>東大吉田塁（酒場の人ではない）先生の、「教員向けChatGPT講座」が分かりやすいと評判に、
<ul>
<li><a href="https://www.youtube.com/live/lwccHzqfuvc?feature=share">https://www.youtube.com/live/lwccHzqfuvc?feature=share</a></li>
</ul>
</li>
<li>Pluginを開発するOSSである、PlugnPlai and LangChainの例
<ul>
<li><a href="https://github.com/edreisMD/plugnplai/blob/master/examples/plugins_step_by_step.ipynb">https://github.com/edreisMD/plugnplai/blob/master/examples/plugins_step_by_step.ipynb</a></li>
</ul>
</li>
<li>HuggingFaceから、自然言語でAgentに指示を出したら画像でも文章でも音声でも出力してくれるモデルを勝手に選んで出力してくれるTransformers  Agent発表、
<ul>
<li><a href="https://huggingface.co/docs/transformers/transformers_agents">https://huggingface.co/docs/transformers/transformers_agents</a></li>
</ul>
</li>
<li>Microsoft社、Sam Altman氏が出資する核融合スタートアップであるHelion Energyと2028に電力購入契約
<ul>
<li><a href="https://www.businessinsider.jp/post-269773">https://www.businessinsider.jp/post-269773</a></li>
<li>加速器で、重水素とHe-3を加速させて衝突時に、磁場で圧縮して、融合させて、膨張の力による磁場の変化から直接電力を（水とか蒸気とかを使わずに）得るという仕組み。</li>
<li>OpenAIはますます、Microsoftと一蓮托生に、、、、</li>
</ul>
</li>
<li>神戸大学、「牧野」先生、不偏分散の自由度がn-1である理由を失念。
<ul>
<li><a href="https://twitter.com/jun_makino/status/1657229042121314304?s=20">https://twitter.com/jun_makino/status/1657229042121314304?s=20</a></li>
<li>牧野先生ご紹介の「美しい導出」<a href="https://manabitimes.jp/math/1205">https://manabitimes.jp/math/1205</a></li>
</ul>
</li>
<li>Scikit-learnの組み込みデータセットから、ボストン住宅価格が、ポリコレのため削除されてた
<ul>
<li><a href="https://twitter.com/tokoroten/status/1394192087453638662?s=20">https://twitter.com/tokoroten/status/1394192087453638662?s=20</a></li>
<li>（授業で使っている人要注意）</li>
</ul>
</li>
<li>Stable Vicuna-13B-4bitがcolabで動作する、ローカルにダウンロードしてWebUIを上げる
<ul>
<li><a href="https://zenn.dev/tatsuromurata/articles/8e523cf2d0c2bc">https://zenn.dev/tatsuromurata/articles/8e523cf2d0c2bc</a></li>
<li><a href="https://note.com/it_navi/n/nceffc6e8df35">https://note.com/it_navi/n/nceffc6e8df35</a></li>
</ul>
</li>
<li>LangChainに、arxiv用のretrieverが追加、Q&amp;Aなどができる
<ul>
<li><a href="https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/arxiv.html">https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/arxiv.html</a></li>
</ul>
</li>
</ul>
<h2 id="section-19">5/8</h2>
<ul>
<li>LlamaIndex 0.6.0 - データに対する新しいクエリインターフェイス
<ul>
<li><a href="https://note.com/npaka/n/n4254fc549dc0">https://note.com/npaka/n/n4254fc549dc0</a></li>
</ul>
</li>
<li>ChatGPT Code Interpreter</li>
<li>Andrew Ngのプロンプトエンジニアリングの講義
<ul>
<li><a href="https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/">https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/</a></li>
</ul>
</li>
<li>Transformerのenc-dec間にinformation bottleneckを入れてVAE的に表現の正則化
<ul>
<li><a href="https://openreview.net/forum?id=6QkjC_cs03X">https://openreview.net/forum?id=6QkjC_cs03X</a></li>
</ul>
</li>
<li>Are Emergent Abilities of Large Language Models a Mirage?
<ul>
<li><a href="https://arxiv.org/abs/2304.15004">https://arxiv.org/abs/2304.15004</a></li>
</ul>
</li>
<li>JDLAでは、「生成AIの利用ガイドライン」
<ul>
<li><a href="https://www.jdla.org/document/?utm_source=prtimes&amp;utm_medium=referral">https://www.jdla.org/document/?utm_source=prtimes&amp;utm_medium=referral</a></li>
</ul>
</li>
<li>LangChainとOpenAIのGymunasiumの連携
<ul>
<li><a href="https://python.langchain.com/en/latest/use_cases/agent_simulations/gymnasium.html">https://python.langchain.com/en/latest/use_cases/agent_simulations/gymnasium.html</a></li>
</ul>
</li>
<li>ディープラーニングによる自然言語処理
<ul>
<li><a href="https://www.amazon.co.jp/dp/4320125029/">https://www.amazon.co.jp/dp/4320125029/</a></li>
</ul>
</li>
<li>Causal Reasoning and Large Language Models: Opening a New Frontier for Causality
<ul>
<li><a href="https://arxiv.org/abs/2305.00050">https://arxiv.org/abs/2305.00050</a></li>
</ul>
</li>
<li>自己アテンション機構をつかって多電子系のシュレディンガー方程式を第一原理的に解く
<ul>
<li><a href="https://arxiv.org/abs/2211.13672">https://arxiv.org/abs/2211.13672</a></li>
</ul>
</li>
<li>OpenLLAMA
<ul>
<li><a href="https://github.com/openlm-research/open_llama">https://github.com/openlm-research/open_llama</a></li>
</ul>
</li>
<li>G.Hintonによる、GAIインタビュー @CNN
<ul>
<li><a href="https://www.youtube.com/watch?v=FAbsoxQtUwM">https://www.youtube.com/watch?v=FAbsoxQtUwM</a></li>
</ul>
</li>
<li>Chatbot Arena: Benchmarking LLMs in the Wild with Elo Ratings
<ul>
<li><a href="https://lmsys.org/blog/2023-05-03-arena/">https://lmsys.org/blog/2023-05-03-arena/</a></li>
</ul>
</li>
<li>TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis
<ul>
<li><a href="https://mathis.petrovich.fr/tmr/">https://mathis.petrovich.fr/tmr/</a></li>
</ul>
</li>
<li>LLMs &amp; Causal Reasoning
<ul>
<li><a href="https://arxiv.org/abs/2305.00050">https://arxiv.org/abs/2305.00050</a></li>
</ul>
</li>
<li>LangChainのv0.0.139からv0.0.151までの差分を整理（もくもく会向け）
<ul>
<li><a href="https://note.com/mahlab/n/ne29d4bfb1d45">https://note.com/mahlab/n/ne29d4bfb1d45</a></li>
</ul>
</li>
<li>LLAMAindexの新しい抽象化API,brand new “router” abstraction in order to build powerful, generalizable, LLM-powered query engines over your data.
<ul>
<li><a href="https://colab.research.google.com/drive/1KH8XtRiO5spa8CT7UrXN54IWdZk3DDxl?usp=sharing">https://colab.research.google.com/drive/1KH8XtRiO5spa8CT7UrXN54IWdZk3DDxl?usp=sharing</a></li>
</ul>
</li>
<li>ホワイトハウスNew Actions to Promote Responsible AI Innovation that Protects Americans’ Rights and  Safety
<ul>
<li><a href="https://www.whitehouse.gov/briefing-room/statements-releases/2023/05/04/fact-sheet-biden-harris-administration-announces-new-actions-to-promote-responsible-ai-innovation-that-protects-americans-rights-and-safety/">https://www.whitehouse.gov/briefing-room/statements-releases/2023/05/04/fact-sheet-biden-harris-administration-announces-new-actions-to-promote-responsible-ai-innovation-that-protects-americans-rights-and-safety/</a></li>
</ul>
</li>
<li>OpenAlpaca, an instruction-following model based on OpenLLaMA
<ul>
<li><a href="https://github.com/yxuansu/OpenAlpaca">https://github.com/yxuansu/OpenAlpaca</a></li>
</ul>
</li>
<li>「LlamaIndex」が0.6.0で大きな変更があったので更新しました。
<ul>
<li><a href="https://note.com/npaka/n/n50475d6c3118">https://note.com/npaka/n/n50475d6c3118</a></li>
</ul>
</li>
<li>ChromaDB Self-Querying Retriever
<ul>
<li><a href="https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/retrievers/examples/chroma_self_query_retriever.ipynb">https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/retrievers/examples/chroma_self_query_retriever.ipynb</a></li>
</ul>
</li>
<li>experimental CodeChain、LangChainの上でPythonを実行できるらしい。
<ul>
<li><a href="https://langchain-ai.github.io/kork/">https://langchain-ai.github.io/kork/</a></li>
</ul>
</li>
<li>Unifying LLM-powered QA Techniques with Routing Abstractions
<ul>
<li><a href="https://betterprogramming.pub/unifying-llm-powered-qa-techniques-with-routing-abstractions-438e2499a0d0">https://betterprogramming.pub/unifying-llm-powered-qa-techniques-with-routing-abstractions-438e2499a0d0</a></li>
</ul>
</li>
<li>PandasAI、またまたpandaベースのチャット解析ツール、OpenAI以外のLLMの使えそう。
<ul>
<li><a href="https://github.com/gventuri/pandas-ai">https://github.com/gventuri/pandas-ai</a></li>
</ul>
</li>
</ul>
<h2 id="section-20">4/10</h2>
<ul>
<li>LLMの倫理的なふるまいをさせるための、マキャベリベンチマーク
<ul>
<li><a href="https://arxiv.org/abs/2304.03279" title="https://arxiv.org/abs/2304.03279">https://arxiv.org/abs/2304.03279</a></li>
</ul>
</li>
<li>LLaMA-Adapter:軽量なLoRAみたいなしくみらしい。
<ul>
<li><a href="https://arxiv.org/abs/2303.16199" title="https://arxiv.org/abs/2303.16199">https://arxiv.org/abs/2303.16199</a></li>
</ul>
</li>
<li>DeepMindから “Formal Algorithms for Transformers”
<ul>
<li><a href="https://arxiv.org/abs/2207.09238" title="https://arxiv.org/abs/2207.09238">https://arxiv.org/abs/2207.09238</a></li>
</ul>
</li>
<li>LLMに対して心理学的な評価（セラピー？）を行う枠組み
<ul>
<li><a href="https://arxiv.org/abs/2207.09238" title="https://arxiv.org/abs/2207.09238">https://arxiv.org/abs/2207.09238</a></li>
</ul>
</li>
<li>リーガルなGPT-4ベースのサービス、Harvey（米ドラマのSUITSの主人公の一人がハーベイ）
<ul>
<li><a href="https://harvey-ai.notion.site/Careers-Harvey-c9e804fe422e4316bdfde9fe74ed6b06" title="https://harvey-ai.notion.site/careers-harvey-c9e804fe422e4316bdfde9fe74ed6b06">https://harvey-ai.notion.site/Careers-Harvey-c9e804fe422e4316bdfde9fe74ed6b06</a></li>
</ul>
</li>
<li>京大２回生の統計力学の期末試験の問題が、論文になった話、
<ul>
<li><a href="https://www.t.u-tokyo.ac.jp/press/pr2023-04-05-001" title="https://www.t.u-tokyo.ac.jp/press/pr2023-04-05-001">https://www.t.u-tokyo.ac.jp/press/pr2023-04-05-001</a></li>
</ul>
</li>
<li>AzureのOpenAI、Embeddingのバージョン２が登場、トークン数が2,048→8,191と激増
<ul>
<li><a href="https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#embeddings-models-1" title="https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#embeddings-models-1">https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#embeddings-models-1</a></li>
</ul>
</li>
<li>MatCha: グラフとかの入力、からも推論やQ&amp;Aができる。 by GoogleAI
<ul>
<li><a href="https://arxiv.org/abs/2212.09662" title="https://arxiv.org/abs/2212.09662">https://arxiv.org/abs/2212.09662</a></li>
</ul>
</li>
<li>gpt4allの公式チャットUIがリリース
<ul>
<li><a href="https://github.com/nomic-ai/gpt4all-ui" title="https://github.com/nomic-ai/gpt4all-ui">https://github.com/nomic-ai/gpt4all-ui</a></li>
</ul>
</li>
<li>MS ResearchのSparks of AGI: early experiments with GPT-4の著者による説明。。
<ul>
<li><a href="https://www.youtube.com/watch?v=qbIk7-JPB2c&amp;t=1023s" title="https://www.youtube.com/watch?v=qbik7-jpb2c&amp;t=1023s">https://www.youtube.com/watch?v=qbIk7-JPB2c&amp;t=1023s</a></li>
</ul>
</li>
</ul>
<h2 id="section-21">4/17</h2>
<ul>
<li>今井むつみ先生の講演「AI時代に必要な学びと教育ー認知科学からの視点」(2023年3月29日)がyoutube配信される
<ul>
<li><a href="https://www.youtube.com/playlist?list=PLMITB-DRUs7N10WLl_4zDUWfBkLd6z_Em" title="https://www.youtube.com/playlist?list=plmitb-drus7n10wll_4zduwfbkld6z_em">https://www.youtube.com/playlist?list=PLMITB-DRUs7N10WLl_4zDUWfBkLd6z_Em</a></li>
</ul>
</li>
<li>DatabircksからDoly2.0がリリース(OSSかつ商用利用可)
<ul>
<li><a href="https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm" title="https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm">https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm</a></li>
</ul>
</li>
<li>CMUの化学者による、LLMを使った合成実験に係る危険性についての露文
<ul>
<li><a href="https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm" title="https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm">https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm</a></li>
</ul>
</li>
<li><a href="https://twitter.com/OpenAI" title="https://twitter.com/openai">@OpenAI</a>の大天才研究者<a href="https://twitter.com/ilyasut" title="https://twitter.com/ilyasut">@ilyasu</a>による、LLMにおけるvisonの重要性と、GPT-4にはvisonも入っているよビデオ
<ul>
<li><a href="https://twitter.com/i/status/1645752089140957187" title="https://twitter.com/i/status/1645752089140957187">https://twitter.com/i/status/1645752089140957187</a></li>
</ul>
</li>
<li>GPT4ALLを使ったApatch2ライセンスのチャットボッドOSSが公開
<ul>
<li><a href="https://github.com/nomic-ai/gpt4all" title="https://github.com/nomic-ai/gpt4all">https://github.com/nomic-ai/gpt4all</a></li>
</ul>
</li>
<li>GPTを用いた触媒開発。ベイズ最適化とLLMを組み合わせて、合成条件を見つける。しかもin context learningを使うので、チューニングも不要！！！（素のGPTでOKということ）。ガウス過程回帰と同程度の性能。逆設計も可能
<ul>
<li><a href="https://arxiv.org/abs/2304.05341v1" title="https://arxiv.org/abs/2304.05341v1">https://arxiv.org/abs/2304.05341v1</a></li>
</ul>
</li>
<li>AlpacaにCoTとStorytellingを強化した、Alpacino30b公開
<ul>
<li><a href="https://huggingface.co/digitous/Alpacino30b/tree/main" title="https://huggingface.co/digitous/alpacino30b/tree/main">https://huggingface.co/digitous/Alpacino30b/tree/main</a></li>
</ul>
</li>
</ul>
<h2 id="section-22">4/24</h2>
<ul>
<li>Microsoft のSemantic KernelのPythonバインディングが発表
<ul>
<li><a href="https://github.com/microsoft/semantic-kernel/blob/main/python/README.md">https://github.com/microsoft/semantic-kernel/blob/main/python/README.md</a></li>
</ul>
</li>
<li>gist tokenによりプロンプトを圧縮する論文(26倍?)
<ul>
<li><a href="https://arxiv.org/abs/2304.08467">https://arxiv.org/abs/2304.08467</a></li>
</ul>
</li>
<li>LLaVA:　Language-and-Vision Asistant、 画像とvicunaをくっつけた
<ul>
<li><a href="https://llava.hliu.cc/">https://llava.hliu.cc/</a></li>
</ul>
</li>
<li>Google ColabでDolly2.0を試す方法
<ul>
<li><a href="https://note.com/npaka/n/nac386bf799b6">https://note.com/npaka/n/nac386bf799b6</a></li>
</ul>
</li>
<li>The Complete Beginners Guide To Autonomous Agents
<ul>
<li><a href="https://www.mattprd.com/p/the-complete-beginners-guide-to-autonomous-agents">https://www.mattprd.com/p/the-complete-beginners-guide-to-autonomous-agents</a></li>
</ul>
</li>
<li>例のCMUの論文：Emergent autonomous scientific research capabilities of large language models
<ul>
<li><a href="https://arxiv.org/abs/2304.05332v1">https://arxiv.org/abs/2304.05332v1</a></li>
</ul>
</li>
<li>シンボリックソルバとLLMを組み合わせるSolving Math Word Problems by Combining Language Models With Symbolic Solvers
<ul>
<li><a href="https://arxiv.org/abs/2304.09102">https://arxiv.org/abs/2304.09102</a></li>
</ul>
</li>
<li>Text2Performer: Text-Driven Human Video Generation
<ul>
<li><a href="https://yumingj.github.io/projects/Text2Performer.html">https://yumingj.github.io/projects/Text2Performer.html</a></li>
</ul>
</li>
<li>Transformerを超えるんじゃないかと言われてる新たな系列モデル（と理解してる）S4とその更なる発展であるH3
<ul>
<li><a href="https://techblog.morphoinc.com/entry/2022/05/24/102648">https://techblog.morphoinc.com/entry/2022/05/24/102648</a></li>
</ul>
</li>
<li>Stability AI	真のOSSのLLM?
<ul>
<li><a href="https://ja.stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models">https://ja.stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models</a></li>
</ul>
</li>
<li>Bing ChatがLaTeXの式を成型できるように進化</li>
<li>Neo4jの知識をエージェントとして、LLMに組み込む話(LangChainの話）
<ul>
<li><a href="https://towardsdatascience.com/implementing-a-sales-support-agent-with-langchain-63c4761193e7">https://towardsdatascience.com/implementing-a-sales-support-agent-with-langchain-63c4761193e7</a></li>
</ul>
</li>
<li>MicrosoftのLow-code LLM、ビジュアルなLLMを使ったアプリ開発環境？
<ul>
<li><a href="https://arxiv.org/abs/2304.08103">https://arxiv.org/abs/2304.08103</a></li>
</ul>
</li>
<li>LLMを使ったアプリ開発で気を付けること（良記事）Building LLM applications for production
<ul>
<li><a href="https://huyenchip.com/2023/04/11/llm-engineering.html">https://huyenchip.com/2023/04/11/llm-engineering.html</a></li>
</ul>
</li>
<li>GoogleのTime-Series Dense Encoder、長いスケールの時系列予測ができるのか？
<ul>
<li><a href="https://ai.googleblog.com/2023/04/recent-advances-in-deep-long-horizon.html">https://ai.googleblog.com/2023/04/recent-advances-in-deep-long-horizon.html</a></li>
</ul>
</li>
<li>GoogleのBardがPythonなどのコード生成ができるようになった
<ul>
<li><a href="https://blog.google/technology/ai/code-with-bard/">https://blog.google/technology/ai/code-with-bard/</a></li>
</ul>
</li>
<li>化学における生成モデルの活用サーベイ？# Generative Models as an Emerging Paradigm in the Chemical Sciences
<ul>
<li><a href="https://pubs.acs.org/doi/10.1021/jacs.2c13467">https://pubs.acs.org/doi/10.1021/jacs.2c13467</a></li>
</ul>
</li>
<li>特別講演「大規模言語モデルの驚異と脅威」、岡崎 直観（東京工業大学情報理工学院 教授）
<ul>
<li><a href="https://youtu.be/PUuk4Cv-ycg">https://youtu.be/PUuk4Cv-ycg</a></li>
</ul>
</li>
<li>ChatGPTでSQL queryを生成する例
<ul>
<li><a href="https://www.promptingguide.ai/applications/coding">https://www.promptingguide.ai/applications/coding</a></li>
</ul>
</li>
<li>LMQL(Language Model Query Language)とLlamaIndexを接続してみる
<ul>
<li><a href="https://note.com/mahlab/n/n34ac7ebf0387">https://note.com/mahlab/n/n34ac7ebf0387</a></li>
</ul>
</li>
<li>**LMQL（Language Model Query Language）**はこのような問題を解決するために開発されている大規模言語モデル（LLM）向けのプログラミング言語です。
<ul>
<li><a href="https://note.com/mahlab/n/n11b15b323c87">https://note.com/mahlab/n/n11b15b323c87</a></li>
</ul>
</li>
<li>ChatGPT + LangChain で、膨大な PDF ドキュメントの内容を爆速で把握する
<ul>
<li><a href="https://qiita.com/hiroki_okuhata_int/items/7102bab7d96eb2574e7d">https://qiita.com/hiroki_okuhata_int/items/7102bab7d96eb2574e7d</a></li>
</ul>
</li>
<li>Generative Agents: Interactive Simulacra of Human Behavior
<ul>
<li><a href="https://arxiv.org/abs/2304.03442">https://arxiv.org/abs/2304.03442</a></li>
</ul>
</li>
<li>物性予測モデルの悪意ある使用を防ぐための論文。Censoring chemical data to mitigate dual use risk
<ul>
<li><a href="https://arxiv.org/abs/2304.10510v1">https://arxiv.org/abs/2304.10510v1</a></li>
</ul>
</li>
<li>LangChainの新機能Contextual Compression Retriever
<ul>
<li><a href="https://note.com/mahlab/n/n7d72e83904cc">https://note.com/mahlab/n/n7d72e83904cc</a></li>
</ul>
</li>
</ul>
<blockquote>
<p>Written with <a href="https://stackedit.io/">StackEdit</a>.</p>
</blockquote>

