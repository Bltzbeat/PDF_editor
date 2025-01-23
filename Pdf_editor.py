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
        
        # Display the current page with clickable elements
        pix = page.get_pixmap()
        img_bytes = pix.tobytes()
        
        # Create columns for image and element info
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(img_bytes, caption=f"Page {page_num + 1}", use_column_width=True)
            
        with col2:
            st.write("Click elements on the page to select them:")
            
            # Get all elements on the page
            text_instances = page.get_text("words")
            images = page.get_images()
            links = page.get_links()
            
            # Create clickable areas for each element
            for i, text in enumerate(text_instances):
                rect = fitz.Rect(text[:4])
                if st.button(f"Text: {text[4][:20]}...", key=f"text_{i}"):
                    st.session_state.selected_element = ("text", i, text)
                    
            for i, img in enumerate(images):
                rect = page.get_image_bbox(img[0])
                if st.button(f"Image {i+1}", key=f"image_{i}"):
                    st.session_state.selected_element = ("image", i, img)
                    
            for i, link in enumerate(links):
                if st.button(f"Link: {link.get('uri', 'No URI')[:20]}...", key=f"link_{i}"):
                    st.session_state.selected_element = ("link", i, link)
        
        # Handle selected element
        if hasattr(st.session_state, 'selected_element'):
            elem_type, elem_idx, elem = st.session_state.selected_element
            
            st.write("\nSelected Element:")
            if elem_type == "text":
                edit_option = st.radio("Choose action:", ["Remove text", "Edit text"])
                if edit_option == "Remove text" and st.button("Remove Selected Text"):
                    rect = fitz.Rect(elem[:4])
                    page.add_redact_annot(rect)
                    page.apply_redactions()
                elif edit_option == "Edit text" and st.button("Edit Selected Text"):
                    new_text = st.text_input("Enter new text:", value=elem[4])
                    rect = fitz.Rect(elem[:4])
                    page.add_redact_annot(rect)
                    page.apply_redactions()
                    page.insert_text(rect.tl, new_text)
                    
            elif elem_type == "image":
                edit_option = st.radio("Choose action:", ["Remove image", "Replace image"])
                if edit_option == "Remove image" and st.button("Remove Selected Image"):
                    rect = page.get_image_bbox(elem[0])
                    page.add_redact_annot(rect)
                    page.apply_redactions()
                elif edit_option == "Replace image":
                    new_image = st.file_uploader("Choose replacement image", type=['png', 'jpg', 'jpeg'])
                    if new_image and st.button("Replace Selected Image"):
                        rect = page.get_image_bbox(elem[0])
                        page.add_redact_annot(rect)
                        page.apply_redactions()
                        page.insert_image(rect, stream=new_image.getvalue())
                        
            elif elem_type == "link":
                edit_option = st.radio("Choose action:", ["Remove link", "Edit link"])
                if edit_option == "Remove link" and st.button("Remove Selected Link"):
                    page.delete_link(elem)
                elif edit_option == "Edit link":
                    new_uri = st.text_input("New URL:", value=elem.get('uri', ''))
                    if st.button("Update Selected Link"):
                        page.delete_link(elem)
                        page.insert_link(elem["rect"], uri=new_uri)
            
            # Refresh the page display after modifications
            pix = page.get_pixmap()
            img_bytes = pix.tobytes()
            with col1:
                st.image(img_bytes, caption=f"Page {page_num + 1} (Updated)", use_column_width=True)
        
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
