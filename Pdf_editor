import PyPDF2
import fitz  # PyMuPDF
import streamlit as st
import os

def edit_pdf(uploaded_file):
    if uploaded_file is None:
        return

    try:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Open PDF with PyMuPDF
        doc = fitz.open("temp.pdf")
        
        # Show available pages
        st.write(f"PDF has {len(doc)} pages.")
        page_num = st.number_input("Enter page number to edit", min_value=1, max_value=len(doc), value=1) - 1

        page = doc[page_num]
        
        # Get page elements
        st.write("\nAvailable elements on this page:")
        action = st.selectbox("What would you like to edit/remove?", 
                            ["Select an action", "Text", "Images", "Links"])
        
        if action == "Text":
            # Handle text editing
            text_instances = page.get_text("words")
            st.write("\nText content on this page:")
            text_options = ["Select text to remove"] + [text[4] for text in text_instances]
            selected_text = st.selectbox("Select text to remove:", text_options)
            
            if selected_text != "Select text to remove" and st.button("Remove Selected Text"):
                text_idx = text_options.index(selected_text) - 1
                text_instance = text_instances[text_idx]
                rect = fitz.Rect(text_instance[:4])
                page.add_redact_annot(rect)
                page.apply_redactions()
        
        elif action == "Images":
            # Handle image removal
            images = page.get_images()
            if images:
                st.write(f"Found {len(images)} images on this page")
                if st.button("Remove all images"):
                    for img in images:
                        rect = page.get_image_bbox(img[0])
                        page.add_redact_annot(rect)
                        page.apply_redactions()
            else:
                st.write("No images found on this page")
        
        elif action == "Links":
            # Handle links
            links = page.get_links()
            if links:
                st.write(f"Found {len(links)} links on this page")
                if st.button("Remove all links"):
                    for link in links:
                        page.delete_link(link)
            else:
                st.write("No links found on this page")
        
        if st.button("Save Changes"):
            output_path = "modified_" + uploaded_file.name
            doc.save(output_path)
            
            # Provide download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download modified PDF",
                    data=file,
                    file_name=output_path,
                    mime="application/pdf"
                )
            
            # Cleanup
            os.remove(output_path)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        if 'doc' in locals():
            doc.close()
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

def main():
    st.title("PDF Editor Tool")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        edit_pdf(uploaded_file)

if __name__ == "__main__":
    main()
