## Workshop

### Step 1: Create a Google Colab Notebook
1. **Open Google Colab**: Go to [Google Colab](https://colab.research.google.com/).
2. **Create a new notebook**: Click on `New Notebook` to start a blank notebook.

### Step 1: Clone the GitHub Repository in Colab
1. **Add a code cell** at the top of the notebook.
2. **Clone the repository** using the following command:
   ```python
   !git clone https://github.com/Maggodbz/sentimaent_sample.git
    ```

### Step 3: Load Packages
1. **Add a code cell** below the previous cell.
2. **Import the necessary packages**:
   ```python
   !pip install torch torchvision torchaudio pytorch_lightning transformers pandas
    ```

### Step 4: Prepare the Data
1. **Add a code cell** below the previous cell.
2. **Run the following code** to load the data:
   ```python
   !python sentimaent_sample/src/preprocess.py
   ```