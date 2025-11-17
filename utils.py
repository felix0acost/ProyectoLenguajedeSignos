import os
# Función para crear la estructura de carpetas
def create_folder_structure(base_path='signs-recogn/dataset'):
   """Crea carpetas train, test y val con subcarpetas para cada letra"""
   folders = ['train', 'test', 'val']
   letters = 'abcdefghijklmnopqrstuvwxyz'
   # Obtener ruta absoluta para Mac
   base_path = os.path.abspath(base_path)
   for folder in folders:
      for letter in letters:
         path = os.path.join(base_path, folder, letter)
         try:
            os.makedirs(path, exist_ok=True)
         except Exception as e:
            print(f"Error creando carpeta {path}: {e}")
   print(f"✓ Estructura de carpetas creada en '{base_path}'")
   return base_path

