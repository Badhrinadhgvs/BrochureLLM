import os
import requests
import json
from bs4 import BeautifulSoup
from openai import OpenAI
import re
import gradio as gr
from IPython.display import display,Markdown

class BrochureLLM:
    def __init__(self, url: str | object):
        self.url = url
        res = requests.get(url)
        self.body = res.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No Title for Webpage"
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
        self.body = soup.body
        self.client =OpenAI(base_url="http://localhost:11434/v1",api_key="ollama")


    def scrap_web(self, url):
        """Scrape title + text + links from a given URL and always return dict"""
        try:
            res = requests.get(url, timeout=10)
            body = res.content
            soup = BeautifulSoup(body, 'html.parser')
            title = soup.title.string if soup.title else "No Title for Webpage"
            text = " ".join([p.get_text() for p in soup.find_all('p')])[:1000]  # limit text
            links = [link.get('href') for link in soup.find_all('a') if link.get('href')]
            return {
                "title": title,
                "url": url,
                "text": text,
                "links": links
            }
        except Exception as e:
            return {
                "title": "Error",
                "url": url,
                "text": str(e),
                "links": []
            }

    def collect_relevant_links(self):
        """Ask LLM to filter relevant links & return JSON"""
        links_sys_prompt = """
        You are an Expert in Brochure Making.
        Your duty is to collect relevant links for Brochure Making.
        You are about to be given a Set of Links. 
        Now choose only the relevant Links for Brochure Making.
        Avoid mailto, license, login, privacy, and irrelevant links.

        Strictly Follow Output Format:
        {
          "links":[
            {"type":"home page","link":"https://something.com"},
            {"type":"another page","link":"https://another.something.com"}
                        {"type":"about page","link":"https://something.com/about"}
          ]
        }
        ONLY return valid JSON. No extra text.
        """

        links_user_prompt = f"Website title: {self.title}\nLinks: {self.links}"

        res = self.client.chat.completions.create(
            model="gpt-oss:20b-cloud",
            messages=[
                {"role": "system", "content": links_sys_prompt},
                {"role": "user", "content": links_user_prompt}
            ]
        )

        data= res.choices[0].message.content
        return data

    def create_brochure(self):
        message =[
            {'role':'system','content':f' You are given with a Website name {self.url}. You will be Provided with the Scraped Data from links generate a Wonderful,Professional Brochure.Strictly Respond in Markdown Only'},
        {'role':'user','content':f"The Scraped Contents of website are \n{self.collect_relevant_links() } and {self.body}"}
        ]
        res = self.client.chat.completions.create(
            model = "gpt-oss:20b-cloud",
            messages=message,
            
        )
        display(Markdown(res.choices[0].message.content))
        
        
def generate_brochure_from_url(url):
    try:
        brochure_bot = BrochureLLM(url)
        message = [
            {'role': 'system', 'content': f'You are given with a Website name {brochure_bot.url}. You will be Provided with the Scraped Data from links generate a Wonderful,Professional Brochure.Strictly Respond in Markdown Only. If possible Provide in a PDF Format'},
            {'role': 'user', 'content': f"The Scraped Contents of website are \n{brochure_bot.collect_relevant_links()} and {brochure_bot.body}"}
        ]
        res = brochure_bot.client.chat.completions.create(
            model="gpt-oss:20b-cloud",
            messages=message,
            stream = True
        )
        out=''
        for chunk in res:
            out += chunk.choices[0].delta.content
            yield out
    except Exception as e:
        return f"Error generating brochure: {str(e)}"

# Gradio Interface
if __name__=="__main__":
    main = gr.Interface(
    fn=generate_brochure_from_url,
    inputs=[gr.Textbox(label="Enter Website URL", placeholder="https://example.com")],
    outputs=gr.Markdown(label="Generated Brochure"),
    title="📄 Brochure Generator",
    description="Enter a website URL to generate a professional brochure using LLM and web scraping.",
    flagging_mode="never")
    main.launch(share=True)