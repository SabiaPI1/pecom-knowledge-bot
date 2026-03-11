import os
import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

from llama_index.core import Document, PromptTemplate, get_response_synthesizer
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from langchain_community.vectorstores import ElasticsearchStore

class SaigaLLM(CustomLLM):
    num_output: int = 512
    model_name: str = "Saiga"
    model: any = None
    tokenizer: any = None
    device: any = None

    def __init__(self, model, tokenizer, device, num_output=512):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_output = num_output

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=self.num_output, model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: any) -> CompletionResponse:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.num_output,
                temperature=0.2, 
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                **kwargs
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "bot\n" in text:
            text = text.split("bot\n")[-1].strip()
        else:
            text = text.replace(prompt, "").strip()
            
        return CompletionResponse(text=text)

class KnowledgeBaseRAG:
    def __init__(self, es_host, es_user, es_password):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используемое устройство: {self.device}")
        
        self.es = Elasticsearch([es_host], 
            basic_auth=(es_user, es_password),
            verify_certs=False,
            request_timeout=60
        )
        
        print("Загрузка модели эмбеддингов...")
        self.embed_model = SentenceTransformer('distiluse-base-multilingual-cased-v2').to(self.device)
        
        self.vector_store = ElasticsearchStore(
            index_name="articles",
            embedding=self.embed_model,
            es_connection=self.es,
            vector_query_field='vector',
            query_field='text',
            distance_strategy='COSINE'
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        print("Загрузка реранкера...")
        self.rerank = SentenceTransformerRerank(top_n=3, model="BAAI/bge-reranker-base")
        
        print("Загрузка LLM Saiga (это займет время)...")
        adapt_model_name = "IlyaGusev/saiga_mistral_7b_lora"
        base_model_name = "Open-Orca/Mistral-7B-OpenOrca"
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapt_model_name, 
            device_map={"": self.device}, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        self.saiga_llm = SaigaLLM(model=self.model, tokenizer=self.tokenizer, device=self.device)
        
        qa_prompt_tmpl_str = """<s>system
Ты — полезный корпоративный ИИ-ассистент базы знаний. Твоя задача — отвечать на вопросы пользователей, используя ТОЛЬКО предоставленную контекстную информацию. Если ответа нет в тексте, честно скажи "Я не нашел ответа в базе знаний". Не придумывай факты.</s>
<s>user
Контекстная информация:
---------------------
{context_str}
---------------------
Запрос: {query_str}
Ответь на запрос, опираясь только на контекст.</s>
<s>bot
"""
        self.qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        
        self.response_synthesizer = get_response_synthesizer(
            llm=self.saiga_llm,
            text_qa_template=self.qa_prompt_tmpl
        )
        
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
            node_postprocessors=[self.rerank]
        )
        print("Система RAG успешно инициализирована!")

    def ask(self, query: str):
        try:
            response = self.query_engine.query(query)
            
            link = "Ссылка не найдена"
            if response.source_nodes and len(response.source_nodes) > 0:
                 node_metadata = response.source_nodes[0].node.metadata
                 link = node_metadata.get("link", link)

            return {
                "answer": str(response),
                "link": link
            }
        except Exception as e:
            return {"answer": f"Произошла ошибка при поиске: {e}", "link": ""}