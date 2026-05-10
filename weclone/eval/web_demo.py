"""
Web Demo for WeClone with quantization support.

This module provides a Gradio-based web interface for chatting with the fine-tuned model.
It properly passes quantization parameters from the configuration file.
"""

import gradio as gr
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

from weclone.utils.config import load_config
from weclone.utils.log import logger


def create_chat_demo():
    """Create a simple chat demo interface using ChatModel with quantization support."""
    
    # Load configuration and create ChatModel
    config = load_config("web_demo")
    config_dict = config.model_dump(mode="json", exclude_none=True)
    
    logger.info(f"Loading model with config: {config_dict}")
    chat_model = ChatModel(config_dict)
    
    def chat(message, history):
        """Chat function for Gradio interface."""
        messages = []
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        
        messages.append({"role": "user", "content": message})
        
        response = ""
        for new_text in chat_model.stream_chat(messages):
            response += new_text
            yield response
    
    def clear_history():
        """Clear chat history and free GPU memory."""
        torch_gc()
        return [], []
    
    with gr.Blocks(title="WeClone Chat") as demo:
        gr.Markdown("# WeClone Chat\n与你的数字分身对话")
        
        chatbot = gr.Chatbot(
            label="对话",
            height=500,
            show_copy_button=True,
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="输入消息",
                placeholder="在这里输入你的消息...",
                scale=9,
            )
            submit_btn = gr.Button("发送", scale=1, variant="primary")
        
        with gr.Row():
            clear_btn = gr.Button("清空对话", variant="secondary")
        
        # Event handlers
        msg.submit(chat, [msg, chatbot], [chatbot])
        submit_btn.click(chat, [msg, chatbot], [chatbot])
        clear_btn.click(clear_history, None, [chatbot, msg])
        
        # Clear input after submit
        msg.submit(lambda: "", None, msg)
        submit_btn.click(lambda: "", None, msg)
    
    return demo


def main():
    """Main entry point for web demo."""
    demo = create_chat_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)


if __name__ == "__main__":
    main()
