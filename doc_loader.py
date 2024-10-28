# Arquivo "doc_loader.py"

#from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import os
import shutil
import pandas as pd
import re

CHROMA_PATH = 'chroma'

# Função para converter datas e criar campos derivados
def process_date_columns(df):
    # Converter para o formato DD/MM/AAAA
    date_columns = ['data_autuacao', 'data_julgamento', 'data_publicacao']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce').dt.strftime('%d/%m/%Y')
        
        # Criar campos derivados para o ano no formato AAAA
        year_col = f"ano_{col}"
        df[year_col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce').dt.year
    
    return df

# Função para limpar o campo ementa
def clean_ementa(ementa):
    if isinstance(ementa, str):
        return ementa.replace('\n', ' ').replace('\r', ' ').strip()  # Remover quebras de linha
    return ementa

# Carregar os dados do arquivo e transformá-los em documentos
def load_documents_from_dataframe(df):
    # Processar as datas e criar os campos derivados de ano
    df = process_date_columns(df)
    
    # Aplicar a limpeza do campo ementa
    df['ementa'] = df['ementa'].apply(clean_ementa)
    
    # Remover a extração de tópicos baseada em 'assunto_descricao' pois a coluna não existe
    # Se desejar utilizar outra coluna para tópicos, substitua aqui

    # Transformar os dados em uma lista de objetos Document
    documents = []
    for index, row in df.iterrows():
        # O conteúdo do documento é apenas o campo 'ementa'
        document_content = row['ementa']
        # Os metadados são os outros campos, garantindo que os valores sejam dos tipos permitidos
        metadata = {
            'document_id': str(index),
            'numero_processo': str(row['numero_processo']).strip(),
            'data_autuacao': str(row['data_autuacao']).strip(),
            'data_julgamento': str(row['data_julgamento']).strip(),
            'data_publicacao': str(row['data_publicacao']).strip(),
            'tempo_de_julgamento': int(row['tempo_de_julgamento']) if pd.notnull(row['tempo_de_julgamento']) else None,
            'classe': str(row['classe']).strip().upper(),
            'ministro_relator': str(row['ministro_relator']).strip().upper(),
            'ministro_revisor': str(row['ministro_revisor']).strip().upper(),
            'ano_data_autuacao': int(row['ano_data_autuacao']) if pd.notnull(row['ano_data_autuacao']) else None,
            'ano_data_julgamento': int(row['ano_data_julgamento']) if pd.notnull(row['ano_data_julgamento']) else None,
            'ano_data_publicacao': int(row['ano_data_publicacao']) if pd.notnull(row['ano_data_publicacao']) else None,
            'source': 'jurisprudencias'
        }
        # Remover campos com valores None
        metadata = {k: v for k, v in metadata.items() if v is not None}

        # Opcionalmente, converter para maiúsculas apenas campos de texto relevantes
        campos_texto = ['classe', 'ministro_relator', 'ministro_revisor']
        for campo in campos_texto:
            if campo in metadata:
                metadata[campo] = metadata[campo].upper()
        
        # Criar um objeto Document para cada linha da planilha
        document = Document(page_content=document_content, metadata=metadata)
        documents.append(document)
    
    return documents

# Função para dividir os documentos
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# Função para adicionar os documentos ao Chroma
def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    
    # Calcular os IDs e adicionar ao DB
    chunks_with_ids = calculate_chunks_ids(chunks)
    
    # Verificar se há IDs duplicados
    ids = [chunk.metadata['id'] for chunk in chunks_with_ids]
    if len(ids) != len(set(ids)):
        duplicates = set([x for x in ids if ids.count(x) > 1])
        print(f"IDs duplicados encontrados: {duplicates}")
        raise ValueError("Há IDs duplicados nos chunks.")
    
    existing_items = db.get(include=[])  # Por padrão os IDs são sempre inclusos
    existing_ids = set(existing_items['ids'])
    print(f'Número de documentos no DB: {len(existing_ids)}')
    
    # Adicionar documentos que não estão no DB.
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata['id'] not in existing_ids]
    
    if new_chunks:
        print(f'👉 Adicionando novos documentos: {len(new_chunks)}')
    
        # Dividir os chunks em lotes de 5461 (tamanho máximo permitido)
        batch_size = 5000
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_ids = [chunk.metadata['id'] for chunk in batch]
            batch_texts = [chunk.page_content for chunk in batch]
            batch_metadatas = [chunk.metadata for chunk in batch]
            db.add_texts(batch_texts, batch_metadatas, ids=batch_ids)
            print(f'Adicionados {len(batch)} documentos ao Chroma (batch {i // batch_size + 1})')
    else:
        print("✅ Nenhum novo documento foi adicionado")

# Função para calcular IDs dos chunks
def calculate_chunks_ids(chunks):
    last_document_id = None
    current_chunk_index = 0
    for chunk in chunks:
        document_id = chunk.metadata.get('document_id')
        if document_id == last_document_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{document_id}:{current_chunk_index}"
        last_document_id = document_id
        chunk.metadata['id'] = chunk_id
    return chunks

# Função principal
def main():
    # Checar se o DB precisa ser limpo
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Carregar o arquivo XLSX
    df = pd.read_excel('docs/jurisprudencias.xlsx')

    # Verificar as colunas disponíveis no DataFrame
    print("Colunas do DataFrame:", df.columns.tolist())

    # Carregar documentos a partir do dataframe
    documents = load_documents_from_dataframe(df)
    
    # Dividir os documentos em chunks
    chunks = split_documents(documents)

    # Adicionar os chunks ao Chroma
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()
