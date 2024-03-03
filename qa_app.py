import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
import tiktoken
import regex
import docx2txt
from io import StringIO

from PIL import Image

#os.environ["OPENAI_API_KEY"]="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"

img = Image.open("img/dl_small.png")
st.set_page_config(page_title="DL",page_icon=img)

# @st.cache_data
# def load_docs(files):
#     st.info("`Reading doc ...`")
#     all_text = ""
#     for file_path in files:
#         file_extension = os.path.splitext(file_path.name)[1]
#         if file_extension == ".pdf":
#             pdf_reader = PyPDF2.PdfReader(file_path)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#             all_text += text
#         elif file_extension == ".txt":
#             stringio = StringIO(file_path.getvalue().decode("utf-8"))
#             text = stringio.read()
#             all_text += text
        
#         else:
#             st.warning('Please provide txt or pdf.', icon="⚠️")
#     return all_text
@st.cache
def load_docs(files):
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        elif file_extension == ".docx":
            text = docx2txt.process(file_path)
            all_text += text
        else:
            st.warning('Please provide txt, pdf, or docx files.', icon="⚠️")
            return ""  # Return empty string or handle appropriately
    return all_text




@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits

# @st.cache_data
# def generate_eval(text, N, chunk):

#     # Generate N questions from context of chunk chars
#     # IN: text, N questions, chunk size to draw question from in the doc
#     # OUT: eval set as JSON list

#     st.info("`Generating sample questions ...`")
#     n = len(text)
#     starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
#     sub_sequences = [text[i:i+chunk] for i in starting_indices]
#     chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
#     eval_set = []
#     for i, b in enumerate(sub_sequences):
#         try:
#             qa = chain.run(b)
#             eval_set.append(qa)
#             st.write("Creating Question:",i+1)
#         except:
#             st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
#     eval_set_full = list(itertools.chain.from_iterable(eval_set))
#     return eval_set_full
@st.cache_data
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list

    st.info("`Generating sample questions ...`")
    n = len(text)
    eval_set = []

    if n < chunk:
        st.warning("Text length is smaller than chunk size.")
        return eval_set

    for i in range(N):
        # Adjust the range if necessary
        if n - chunk <= 0:
            starting_index = 0
        else:
            starting_index = random.randint(0, n - chunk)

        b = text[starting_index: starting_index + chunk]
        try:
            chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creating Question:", i + 1)
        except Exception as e:
            st.warning(f'Error generating question {i+1}: {str(e)}', icon="⚠️")

    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full



# ...

def main():
    
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
        <p>Made by doAZ</p>
    </div>
    """

    st.markdown(foot, unsafe_allow_html=True)
    
    # Add custom CSS
    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
            
           
       
            
          

        </style>
        """,
        unsafe_allow_html=True,
    )
    # st.sidebar.image("img/logo1.png")


    st.image("img/logo2.png")
    
    
    
    
    st.sidebar.title("Menu")
    language = st.text_input(
            '언어 선택', value="", placeholder="한국어 \"k\", 영어 \"e\"")
    language = st.sidebar.radio(
        "Choose Language/언어 선택", ["English/영어", "Korean/한국인"])
    
    embedding_option = "OpenAI Embeddings"
    
    #embedding_option = st.sidebar.radio(
        #"Choose Embeddings", ["OpenAI Embeddings", "HuggingFace Embeddings(slower)"])

     
    retriever_type = st.sidebar.selectbox(
        "Choose Retriever", ["SIMILARITY SEARCH", "SUPPORT VECTOR MACHINES"])

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

    if 'openai_api_key' not in st.session_state:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        #language = st.text_input(
            #'언어 선택', value="", placeholder="한국어 \"k\", 영어 \"e\"")
        #openai_api_key = st.text_input(
            #'Please enter your OpenAI API key or [get one here](https://platform.openai.com/account/api-keys)', value="", placeholder="Enter the OpenAI API key which begins with sk-")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            #warning_text = 'Please enter your OpenAI API key. Get yours from here: [link](https://platform.openai.com/account/api-keys)'
            #warning_html = f'<span>{warning_text}</span>'
            #st.markdown(warning_html, unsafe_allow_html=True)
            return
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    # uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
    #                                   "pdf", "txt","docx"], accept_multiple_files=True)
    # uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=["pdf", "txt", "docx"], accept_multiple_files=True, type=["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"])
    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=["pdf", "txt", "docx"], accept_multiple_files=True)


    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1500,
                             overlap=100, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")

        # Embed using OpenAI embeddings
            # Embed using OpenAI embeddings or HuggingFace embeddings
        if embedding_option == "OpenAI Embeddings":
            embeddings = OpenAIEmbeddings()
        elif embedding_option == "HuggingFace Embeddings(slower)":
            # Replace "bert-base-uncased" with the desired HuggingFace model
            embeddings = HuggingFaceEmbeddings()

        retriever = create_retriever(embeddings, splits, retriever_type)


        # Initialize the RetrievalQA chain with streaming output
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat_openai = ChatOpenAI(
            streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)
        qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

        # Check if there are no generated question-answer pairs in the session state
        if 'eval_set' not in st.session_state:
            # Use the generate_eval function to generate question-answer pairs
            num_eval_questions = 30  # Number of question-answer pairs to generate
            st.session_state.eval_set = generate_eval(
                loaded_text, num_eval_questions, 7000)

       # Display the question-answer pairs in the sidebar with smaller text
        for i, qa_pair in enumerate(st.session_state.eval_set):
            #st.sidebar.markdown(
             #   f"""
              #  <div class="css-card">
               # <span class="card-tag">Question {i + 1}</span>
                #    <p style="font-size: 12px;">{qa_pair['question']}</p>
                 #   <p style="font-size: 12px;">{qa_pair['answer']}</p>
                #</div>
                #""",
                #unsafe_allow_html=True,
            #)
            st.sidebar.markdown(
                f"""
                <div class="css-card" style="background-color: #001f3f; color: white;">
                <span class="card-tag">Question {i + 1}</span>
                    <p style="font-size: 12px;">{qa_pair['question']}</p>
                    <p style="font-size: 12px;">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # <h4 style="font-size: 14px;">Question {i + 1}:</h4>
            # <h4 style="font-size: 14px;">Answer {i + 1}:</h4>
        st.write("Ready to answer questions.")

        def find_image(user_question):
            import openai
            import json
            #st.info("`finding relevant image ...`")
            

            # Securely load your OpenAI API key (don't expose it directly in code)
            openai.api_key = os.environ["OPENAI_API_KEY"]  # Replace with your actual key

            # Load the JSON file containing image captions
            # with open("image.json", "r") as f:
            #     image_data = json.load(f)
                #st.info("`loading image data...`")

            # Define the question
            question = user_question
            Exactly_matched = False
            # prompt 
##############################################################################
            #FOR EXACT MATCH WITH QUESTION
            # for image_info in image_data:
            #     caption = image_info["imageCaption"]
            #     korean_caption = image_info["KoreanimageCaption"]
            #     image_path = image_info["imagePath"]
                #question_list = image_info["questions"]

                
            # for q in image_info["questions"]: 
            prompt = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": f"If question1 matches exactly with question 2,then just reply with Yes otherwise No\n\nQuestion1 '{user_question}'?\n\nQuestion2 : "},
                ]
            

            
                #prompt = f"does the caption best matches the question, if yes then just say yes otherwise no,  question: '{question}'?\n\nCaption: {caption}"

                # Send the prompt to OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.5,
                )
            #st.info(f"`Response ..."+str(response.choices[0].message.content.split("\n")[-1]))
        
            #st.info(f"`Response ..."+str(response.choices[0].message.content.split("\n")[-1]))
            # Check if OpenAI's response matches the caption
            if response.choices[0].message.content.split("\n")[-1] == "Yes":
                Exactly_matched = True
                # Match found, save image path and caption
                # matched_image_path = image_path
                # matched_caption = caption
                # korean_matched_caption = korean_caption
                #st.info("found matching image...")
            
                #st.image(matched_image_path, caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                if language == "English/영어":
                    #st.image(matched_image_path, caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                    #st.write(matched_caption)
                    # Create columns for layout
                    col1, col2 = st.columns(2)

                    # Display image in first column with appropriate width and caption text size
                    # with col1:
                    #     st.image(
                    #         matched_image_path,
                    #         caption=None,
                    #         width=300,
                    #         use_column_width=None,  # Ensure image respects width in pixels
                    #         clamp=False,
                    #         channels="RGB",
                    #         output_format="auto",
                    #     )

                    # Display caption in second column with small text size and right alignment
                    # with col2:
                    #     st.markdown(
                    #         f"<p style='font-size: small; text-align: left;'>{matched_caption}</p>",
                    #         unsafe_allow_html=True,
                    #     )
                    # st.write(" ")

                    #col1, mid, col2 = st.beta_columns([1,1,20])
                    #with col1:
                        #st.image(matched_image_path, width=300, channels="RGB", output_format = "auto")
                    #with col2:
                    
                        #st.write(matched_caption)

                elif language == "Korean/한국인":
                    #st.image(matched_image_path, caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                    col1, col2 = st.columns(2)

                    # Display image in first column with appropriate width and caption text size
                    # with col1:
                    #     st.image(
                    #         matched_image_path,
                    #         caption=None,
                    #         width=300,
                    #         use_column_width=None,  # Ensure image respects width in pixels
                    #         clamp=False,
                    #         channels="RGB",
                    #         output_format="auto",
                    #     )

                    # Display caption in second column with small text size and right alignment
                    # with col2:
                    #     st.markdown(
                    #         f"<p style='font-size: small; text-align: left;'>{korean_matched_caption}</p>",
                    #         unsafe_allow_html=True,
                    #     )
                    # st.write(" ")
            
            
            






        #######################################################

                
            
            if Exactly_matched == False:
                # for image_info in image_data:
                #     caption = image_info["imageCaption"]
                #     korean_caption = image_info["KoreanimageCaption"]
                #     image_path = image_info["imagePath"]
                

                #     prompt = [
                #             {"role": "system", "content": "You are a helpful assistant"},
                #             {"role": "user", "content": f"provide a short caption for the image that should match exactly with this question,  question: '{question}'?"},
                #             ]
                #     response = openai.ChatCompletion.create(
                #         model="gpt-3.5-turbo",
                #         messages=prompt,
                #         max_tokens=150,
                #         n=1,
                #         stop=None,
                #         temperature=0.5,
                #      )
                #     caption1 = response.choices[0].message.content.split("\n")[-1]
                


                

                    # Create the prompt for OpenAI
                
                    prompt = [
                            {"role": "system", "content": "You are a helpful assistant"},
                            # {"role": "user", "content": f"Do these captions exactly matches with each other, if yes then just say yes otherwise no,Caption1 '{caption1}'?\n\nCaption2: {caption}"},
                            ]
                

                
                    #prompt = f"does the caption best matches the question, if yes then just say yes otherwise no,  question: '{question}'?\n\nCaption: {caption}"

                    # Send the prompt to OpenAI
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=prompt,
                        max_tokens=150,
                        n=1,
                        stop=None,
                        temperature=0.5,
                        )
                    #st.info(f"`Response ..."+str(response.choices[0].message.content.split("\n")[-1]))
                

                    # Check if OpenAI's response matches the caption
                    if response.choices[0].message.content.split("\n")[-1] == "Yes":
                        # Match found, save image path and caption
                        # matched_image_path = image_path
                        # matched_caption = caption
                        # korean_matched_caption = korean_caption
                        #st.info("found matching image...")
                    
                        #st.image(matched_image_path, caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                        if language == "English/영어":
                            #st.image(matched_image_path, caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                            #st.write(matched_caption)
                            # Create columns for layout
                            col1, col2 = st.columns(2)

                            # Display image in first column with appropriate width and caption text size
                            # with col1:
                            #     st.image(
                            #         matched_image_path,
                            #         caption=None,
                            #         width=300,
                            #         use_column_width=None,  # Ensure image respects width in pixels
                            #         clamp=False,
                            #         channels="RGB",
                            #         output_format="auto",
                            #     )

                            # # Display caption in second column with small text size and right alignment
                            # with col2:
                            #     st.markdown(
                            #         f"<p style='font-size: small; text-align: left;'>{matched_caption}</p>",
                            #         unsafe_allow_html=True,
                            #     )
                            # st.write(" ")
        
                            #col1, mid, col2 = st.beta_columns([1,1,20])
                            #with col1:
                                #st.image(matched_image_path, width=300, channels="RGB", output_format = "auto")
                            #with col2:
                            
                                #st.write(matched_caption)

                        elif language == "Korean/한국인":
                            #st.image(matched_image_path, caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                            col1, col2 = st.columns(2)

                            # Display image in first column with appropriate width and caption text size
                            # with col1:
                            #     st.image(
                            #         matched_image_path,
                            #         caption=None,
                            #         width=300,
                            #         use_column_width=None,  # Ensure image respects width in pixels
                            #         clamp=False,
                            #         channels="RGB",
                            #         output_format="auto",
                            #     )

                            # Display caption in second column with small text size and right alignment
                            # with col2:
                            #     st.markdown(
                            #         f"<p style='font-size: small; text-align: left;'>{korean_matched_caption}</p>",
                            #         unsafe_allow_html=True,
                            #     )
                            # st.write(" ")
            
                    #st.info("No match found...")
                    # Iterate through the image data
                






            
                    

                    # Save the matched info (replace with your preferred saving method)
                    #with open("matched_images.txt", "a") as f:
                    #f.write(f"Image Path: {matched_image_path}\nCaption: {matched_caption}\n\n")

                    # Print the matched info for confirmation
                    
            
                    #print(f"Matching image: {matched_image_path}")
                    #print(f"Matching caption: {matched_caption}")

                    # (Optional) You can further process the image here, for example:
                    # - Display the image using an image viewer library
                    # - Download the image to a specific location

                    # If no match found, print a message
                #else:
                    #print("No matching caption found.")
        def does_user_want_to_show(user_question):
            
            keywords = ["콘크리트 펌프카 안전점검표 보여줘","show","display","보여주다","표시하다","보여줘","안전 체크리스트","체크리스트"]
            for keyword in keywords:
                if keyword.lower() in user_question.lower():
                    return True
                    break
            return False
        
        def replace_korean_words(text):
            replacements = {
                "자바라": "호스",
                "공구리":"콘크리트",
            }
            for word, replacement in replacements.items():
                
                text = text.replace(word, replacement)

            return text
        

            

  


            
        
            

        # MAIN  Question and answering
        user_question = st.text_input("Enter your question:")
        if user_question:


            if language == "Korean/한국인":
                
                replaced_text = replace_korean_words(user_question)
                
                show_something = does_user_want_to_show(user_question)
                if show_something:
                    st.info("`관련 이미지 찾기 ...`")
                    st.write(" ")
                    st.write(" ")
                    keywords = ["콘크리트 펌프카 안전점검표 보여줘"]
                    for keyword in keywords:
                        if keyword.lower()  == user_question.lower():
                            st.image(
                                "./images/image0.png",
                                caption="Concrete pump car safety checklist.",
                                width=350,
                                use_column_width=None,  # Ensure image respects width in pixels
                                clamp=False,
                                channels="RGB",
                                output_format="auto",
                            )
                            break
                        else:
                            find_image(user_question)
                else:
                    answer = qa.run(replaced_text)
                    st.write("Answer:", answer)
                    #st.write("Answer:", answer)
                    st.write(" ")
                    st.write(" ")
                    find_image(user_question)
                
                

            elif language == "English/영어":
                show_something = does_user_want_to_show(user_question)
                if show_something:
                    st.info("`finding relevant image ...`")
                    st.write(" ")
                    st.write(" ")
                    keywords = ["콘크리트 펌프카 안전점검표 보여줘"]
                    for keyword in keywords:
                        if keyword.lower()  == user_question.lower():
                            st.image(
                                "./images/image0.png",
                                caption="Concrete pump car safety checklist.",
                                width=350,
                                use_column_width=None,  # Ensure image respects width in pixels
                                clamp=False,
                                channels="RGB",
                                output_format="auto",
                            )
                            break
                        else:
                            find_image(user_question)
                else:
                    answer = qa.run(user_question)
                    st.write("Answer:", answer)
                    st.write(" ")
                    st.write(" ")
                    find_image(user_question)
            #else:
                #st.write("Select a language.")
        
            
            #answer = qa.run(user_question)
            #st.write("Answer:", answer)


if __name__ == "__main__":
    main()
