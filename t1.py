import zipfile

def zip_file(input_file_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_file_path)

# Example usage:
zip_file('india_global_model.pkl', 'india_global_model.zip')
