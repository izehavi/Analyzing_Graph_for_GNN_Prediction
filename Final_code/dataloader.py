import pandas as pd
from datetime import timedelta

class DataLoader:
    """
    Classe pour charger et préparer les données d'entraînement, de validation et de test.
    """

    def __init__(self, path_train: str, path_test: str, **kwargs):
        """
        Initialise la classe avec les chemins des fichiers de données et les configurations.

        Args:
            path_train (str): Chemin vers le fichier CSV des données d'entraînement.
            path_test (str): Chemin vers le fichier CSV des données de test.
        """
        self.path_train = path_train
        self.path_test = path_test

        # Charger les données d'entraînement et de test
        self.df_train_original = pd.read_csv(path_train)
        self.df_test = pd.read_csv(path_test)
        
        # Extraire les données d'entraînement
        self.df_train = self.extract_dataframe(self.df_train_original, **kwargs)
        
        self.nodes_dataframe = self.build_nodes_dataframe(self.df_train, "Region")

        
        
        
    def extract_dataframe(self, df: pd.DataFrame,  **kwargs) :
        """
        Filters a DataFrame based on a date range.

        Args:
            df (pd.DataFrame): Input DataFrame containing a 'date' column.
            start_date (str, optional): Start date (inclusive) in 'YYYY-MM-DD' format. Defaults to the earliest date in the DataFrame.
            end_date (str, optional): End date (exclusive) in 'YYYY-MM-DD' format. Defaults to the latest date in the DataFrame.

        Returns:
            pd.DataFrame : A new DataFrame containing only the rows within the specified date range.
        """
        # get start_date and end_date from kwargs
        start_date = kwargs.get("start_date", None)
        end_date = kwargs.get("end_date", None)
        
        # Ensure the 'date' column exists
        assert 'date' in df.columns, "The DataFrame must contain a 'date' column!"
        
        # Create a copy of the DataFrame to avoid modifying the original
        extracted_df = df.copy()

        # Set default values for start_date and end_date if they are not provided
        if start_date is None:
            start_date = extracted_df['date'].min()  # Earliest date in the DataFrame
        if end_date is None:
            end_date = extracted_df['date'].max()   # Latest date in the DataFrame

        # Apply the date filter
        mask = (extracted_df['date'] >= start_date) & (extracted_df['date'] < end_date)
        extracted_df = extracted_df[mask].reset_index(drop=True)

        return extracted_df
    
    def build_nodes_dataframe(self, df: pd.DataFrame, region_column: str):
        """
        Creating a dictionnary to encode the node ids ands add it dataframe
        """
        # check if the region column is in the dataframe
        assert region_column in df.columns, f"La colonne '{region_column}' est absente du DataFrame !"
        
        # Create a copy of the DataFrame to avoid modifying the original
        nodes_df = df.copy()
        
        # Extract the unique nodes
        region_ids = nodes_df[region_column].unique()
        # Create a dictionary to map the region names to integer node ids
        region_to_id = {region: idx for idx, region in enumerate(region_ids)}
        
        # Add a new column to the DataFrame with the integer node ids
        nodes_df['node_id'] = nodes_df[region_column].map(region_to_id)
        
        # For each region, create a dictionnary with the corresponding dataframe
        nodes_dict = {
        str(region_to_id[region]): nodes_df[nodes_df[region_column] == region].reset_index(drop=True)
        for region in region_ids}
        return nodes_dict
        
        
        
        
        