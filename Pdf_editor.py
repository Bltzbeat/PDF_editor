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
        
        # Display the current page
        pix = page.get_pixmap()
        img_bytes = pix.tobytes()
        st.image(img_bytes, caption=f"Page {page_num + 1}", use_column_width=True)
        
        # Get page elements
        st.write("\nSelect elements to edit/remove from this page:")
        action = st.selectbox("What would you like to edit/remove?", 
                            ["Select an action", "Text", "Images", "Links"])
        
        if action == "Text":
            # Handle text editing
            text_instances = page.get_text("words")
            if text_instances:
                st.write("\nSelect text to edit or remove:")
                text_options = ["Select text"] + [text[4] for text in text_instances]
                selected_text = st.selectbox("Select text:", text_options)
                
                if selected_text != "Select text":
                    edit_option = st.radio("Choose action:", ["Remove text", "Edit text"])
                    
                    if edit_option == "Remove text" and st.button("Remove Selected Text"):
                        text_idx = text_options.index(selected_text) - 1
                        text_instance = text_instances[text_idx]
                        rect = fitz.Rect(text_instance[:4])
                        page.add_redact_annot(rect)
                        page.apply_redactions()
                    
                    elif edit_option == "Edit text" and st.button("Edit Selected Text"):
                        text_idx = text_options.index(selected_text) - 1
                        text_instance = text_instances[text_idx]
                        new_text = st.text_input("Enter new text:", value=selected_text)
                        rect = fitz.Rect(text_instance[:4])
                        page.add_redact_annot(rect)
                        page.apply_redactions()
                        page.insert_text(rect.tl, new_text)
                    
                    # Refresh the page display
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes()
                    st.image(img_bytes, caption=f"Page {page_num + 1} (Updated)", use_column_width=True)
            else:
                st.write("No text found on this page")
        
        elif action == "Images":
            # Handle image editing/removal
            images = page.get_images()
            if images:
                st.write(f"Found {len(images)} images on this page")
                edit_option = st.radio("Choose action:", ["Remove images", "Replace images"])
                
                if edit_option == "Remove images" and st.button("Remove all images"):
                    for img in images:
                        rect = page.get_image_bbox(img[0])
                        page.add_redact_annot(rect)
                        page.apply_redactions()
                
                elif edit_option == "Replace images":
                    new_image = st.file_uploader("Choose replacement image", type=['png', 'jpg', 'jpeg'])
                    if new_image and st.button("Replace images"):
                        for img in images:
                            rect = page.get_image_bbox(img[0])
                            page.add_redact_annot(rect)
                            page.apply_redactions()
                            page.insert_image(rect, stream=new_image.getvalue())
                
                # Refresh the page display
                pix = page.get_pixmap()
                img_bytes = pix.tobytes()
                st.image(img_bytes, caption=f"Page {page_num + 1} (Updated)", use_column_width=True)
            else:
                st.write("No images found on this page")
        
        elif action == "Links":
            # Handle links editing
            links = page.get_links()
            if links:
                st.write(f"Found {len(links)} links on this page")
                edit_option = st.radio("Choose action:", ["Remove links", "Edit links"])
                
                if edit_option == "Remove links" and st.button("Remove all links"):
                    for link in links:
                        page.delete_link(link)
                
                elif edit_option == "Edit links":
                    for i, link in enumerate(links):
                        st.write(f"Link {i+1}: {link.get('uri', 'No URI')}")
                        new_uri = st.text_input(f"New URL for link {i+1}:", value=link.get('uri', ''))
                        if st.button(f"Update link {i+1}"):
                            page.delete_link(link)
                            page.insert_link(link["rect"], uri=new_uri)
                
                # Refresh the page display
                pix = page.get_pixmap()
                img_bytes = pix.tobytes()
                st.image(img_bytes, caption=f"Page {page_num + 1} (Updated)", use_column_width=True)
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
