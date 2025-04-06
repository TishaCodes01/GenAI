from langchain_groq import ChatGroq 
from langchain.schema import SystemMessage 
from langchain.chains import LLMChain 
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder 

import warnings
warnings.filterwarnings('ignore')

class MedBot:
    def __init__(self, api_key="secret_key", model_name="llama-3.3-70b-versatile", temperature=0.3, top_p=0.8):
        """
        Initializes the MedBot chatbot with Groq API and conversation memory.
        """
        self.llm = ChatGroq(
            temperature=temperature,
            groq_api_key=api_key,
            model_name=model_name,
            top_p=top_p
        )

        self.system_prompt = """
            You are MedBot, an AI-powered healthcare assistant. Your role is to provide accurate and reliable medical information.
            Guidelines:
            1. Do not provide information outside the medical field. If asked, reply that you are a medical chatbot and also remember all previous chats with all domain.
            2. Answer only healthcare-related questions.
            3. Use verified medical sources.
            4. Explain complex medical terms in an easy-to-understand way.
            5. Do not provide personal diagnoses or prescribe medications.
            6. If the question requires emergency help, advise the user to consult a medical professional immediately.
            7. Maintain a polite and professional tone at all times.
            8. Avoid unnecessary information or long explanations.
            9. If the user asks for more details, provide them upon request.
        """

        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}")
        ])

        self.chat_chain = LLMChain(llm=self.llm, prompt=self.prompt_template, memory=self.memory)

    def generate_response(self, user_input):
        response = self.chat_chain.run(user_input)
        return response
