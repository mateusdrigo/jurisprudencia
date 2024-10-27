# Arquivo "app.py"

import re
import streamlit as st
import google.generativeai as genai
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from PIL import Image

# App title
st.set_page_config(page_title="Jurisprudência do STM", layout="wide")

# Adicionando o logo do STM
logo = Image.open('logo.png')
st.image(logo, width=200)

# Título principal
st.title("Jurisprudência do STM")

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Você é um assistente especializado em jurisprudências do Superior Tribunal Militar (STM).
A Justiça Militar da União (JMU) é um dos ramos do Poder Judiciário brasileiro, sendo especializada no julgamento de crimes militares. Está dividida em 12 Circunscrições Judiciárias Militares (CJM), cada uma abrigando uma ou mais Auditorias Militares, que atuam como órgãos de 1ª Instância.
Os recursos das decisões de primeira instância são encaminhados diretamente para o Superior Tribunal Militar (STM), que atua como instância final.

{context}

---

**Pergunta:** {question}

**Instruções:**
- Forneça respostas claras e objetivas.
- Se a pergunta for sobre o resultado ou a decisão de um processo, busque a informação na ementa do processo.
- Se a resposta envolver uma lista de processos, forneça os números dos processos.
- Se a pergunta for sobre quantidade, forneça o número exato com a devida formatação.
- Se as informações não estiverem disponíveis, pesquise nos sites https://www.stm.jus.br/, https://www.stm.jus.br/o-stm-stm/institucional, https://www.stm.jus.br/o-stm-stm/composicao-corte-2, https://www.stm.jus.br/o-stm-stm/primeira-instancia.
- Se mesmo assim as informações não estiverem disponíveis, pesquise na Internet. 

**Por favor, formate sua resposta usando Markdown.**
"""

def configure_genai():
    google_api_key = st.secrets["google"]["api_key"]
    genai.configure(api_key=google_api_key)
    genai.GenerationConfig(temperature=0.7)

configure_genai()

def generate_response(prompt_input):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Definir filtros com base na pergunta do usuário
    filters = {}
    prompt_lower = prompt_input.lower()

    # Inicializar flags para determinar o tipo de pergunta
    requires_count = False
    requires_field_retrieval = False
    requires_average = False
    field_to_retrieve = None

    # Analisar a pergunta do usuário
    if re.search(r'quantos processos', prompt_lower):
        requires_count = True

        if 'ministro artur vidigal de oliveira' in prompt_lower and 'relator' in prompt_lower:
            filters['ministro_relator'] = "ARTUR VIDIGAL DE OLIVEIRA"
        elif 'ministro artur vidigal de oliveira' in prompt_lower and 'revisor' in prompt_lower:
            filters['ministro_revisor'] = "ARTUR VIDIGAL DE OLIVEIRA"
        elif 'data de julgamento' in prompt_lower or 'julgados em' in prompt_lower:
            # Extrair o ano
            match = re.search(r'\b(\d{4})\b', prompt_lower)
            if match:
                ano = int(match.group(1))
                filters['ano_data_julgamento'] = ano
        elif 'autuados em' in prompt_lower or 'data de autuação' in prompt_lower:
            match = re.search(r'\b(\d{4})\b', prompt_lower)
            if match:
                ano = int(match.group(1))
                filters['ano_data_autuacao'] = ano
        elif 'classe' in prompt_lower:
            match = re.search(r'classe\s+(.*?)(?:\?|$)', prompt_input, re.IGNORECASE)
            if match:
                classe = match.group(1).strip().upper()
                filters['classe'] = classe
        elif 'relatoria do ministro' in prompt_lower or 'relatoria do' in prompt_lower:
            match = re.search(r'relatoria do ministro\s+(.*?)(?:\?|$)', prompt_input, re.IGNORECASE)
            if not match:
                match = re.search(r'relatoria do\s+(.*?)(?:\?|$)', prompt_input, re.IGNORECASE)
            if match:
                ministro = match.group(1).strip().upper()
                filters['ministro_relator'] = ministro
        elif 'assunto' in prompt_lower:
            match = re.search(r'assunto\s+(.*?)(?:\?|$)', prompt_input, re.IGNORECASE)
            if match:
                assunto = match.group(1).strip().upper()
                filters['assunto_principais'] = {'$contains': assunto}

    elif re.search(r'qual a (classe|decisão|ementa) do processo', prompt_lower):
        requires_field_retrieval = True
        # Extrair o número do processo
        match = re.search(r'processo\s+(\d+)', prompt_input)
        if match:
            numero_processo = match.group(1).strip()
            filters['numero_processo'] = numero_processo
            field_match = re.search(r'qual a (\w+) do processo', prompt_lower)
            if field_match:
                field_to_retrieve = field_match.group(1).lower()
                # Mapear os nomes dos campos para as chaves dos metadados
                field_mapping = {
                    'classe': 'classe',
                    'decisão': 'decisao',  # Certifique-se de que 'decisao' está nos metadados
                    'ementa': 'ementa'
                }
                field_to_retrieve = field_mapping.get(field_to_retrieve)
    
    elif re.search(r'média de tempo de julgamento', prompt_lower):
        requires_average = True
        # Extrair a classe
        match = re.search(r'classe\s+"?(.*?)"?$', prompt_input, re.IGNORECASE)
        if match:
            classe = match.group(1).strip().upper()
            filters['classe'] = classe

    else:
        # Consulta sem filtros específicos
        results = db.similarity_search_with_score(prompt_input, k=5)
        documents = [doc for doc, _score in results]

    # Processar a consulta conforme o tipo
    if requires_count:
        # Realizar a contagem de documentos que correspondem aos filtros
        print(f"Filtros aplicados para contagem: {filters}")
        results = db.get(where=filters, include=['metadatas'])
        count = len(results['metadatas'])
        response = f"{count} processos."
        return response

    elif requires_field_retrieval and field_to_retrieve:
        # Recuperar o campo específico do documento correspondente
        print(f"Filtros aplicados para recuperação de campo: {filters}")
        results = db.get(where=filters, include=['metadatas', 'documents'])
        if results['metadatas']:
            metadata = results['metadatas'][0]
            if field_to_retrieve in metadata:
                field_value = metadata[field_to_retrieve]
                response = f"O {field_to_retrieve} do processo {filters['numero_processo']} é: {field_value}"
            else:
                response = f"O campo '{field_to_retrieve}' não está disponível para o processo {filters['numero_processo']}."
        else:
            response = f"O processo {filters['numero_processo']} não foi encontrado."
        return response

    elif requires_average:
        # Calcular a média do tempo_de_julgamento para a classe especificada
        print(f"Filtros aplicados para cálculo de média: {filters}")
        results = db.get(where=filters, include=['metadatas'])
        tempos = [meta['tempo_de_julgamento'] for meta in results['metadatas'] if 'tempo_de_julgamento' in meta]
        if tempos:
            media = sum(tempos) / len(tempos)
            response = f"A média de tempo de julgamento dos processos da classe \"{filters['classe']}\" é de {media:.2f} dias."
        else:
            response = f"Não há dados disponíveis para calcular a média de tempo de julgamento da classe \"{filters['classe']}\"."
        return response

    else:
        # Consulta padrão usando o modelo de linguagem
        if not filters:
            # Busca semântica
            results = db.similarity_search_with_score(prompt_input, k=5)
            documents = [doc for doc, _score in results]
        else:
            # Aplicar filtros e obter documentos
            print(f"Filtros aplicados: {filters}")
            results = db.get(include=['metadatas', 'documents'], where=filters)
            documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(results['documents'], results['metadatas'])]

        # Construir o contexto para o modelo
        context_text = "\n\n---\n\n".join([f"Ementa: {doc.page_content}\nMetadados: {doc.metadata}" for doc in documents])

        # Preparar o prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        input_prompt = prompt_template.format(context=context_text, question=prompt_input)

        # Gerar a resposta usando o modelo
        model = genai.GenerativeModel('gemini-1.5-flash')
        output = model.generate_content(input_prompt)

        # Extrair a resposta
        full_response = ''
        for item in output._result.candidates[0].content.parts:
            full_response += item.text

        return full_response

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Eu sou o assistente de jurisprudências do STM. Como posso ajudar?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = generate_response(prompt)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
