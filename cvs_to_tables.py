import pandas as pd
import matplotlib.pyplot as plt

def csv_to_table_image(csv_file, output_file):
    # Lê o CSV
    df = pd.read_csv(csv_file)

    # Cria figura
    fig, ax = plt.subplots(figsize=(10, len(df)*0.6))  # ajusta altura conforme nº linhas
    ax.axis("off")

    # Cria tabela
    tabela = ax.table(cellText=df.values,
                      colLabels=df.columns,
                      cellLoc="center",
                      loc="center")

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1.2, 1.2)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    arquivos = [
        ("classificacao_holdout.csv", "classificacao_holdout.png"),
        ("classificacao_crossval.csv", "classificacao_crossval.png"),
        ("regressao_holdout.csv", "regressao_holdout.png"),
        ("regressao_crossval.csv", "regressao_crossval.png"),
    ]

    for csv_file, img_file in arquivos:
        csv_to_table_image(csv_file, img_file)
        print(f"✅ {img_file} gerada a partir de {csv_file}")
