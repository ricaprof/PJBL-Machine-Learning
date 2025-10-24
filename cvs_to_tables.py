import pandas as pd
import matplotlib.pyplot as plt

def csv_to_table_image(csv_file, output_file):
    # Lê o CSV
    df = pd.read_csv(csv_file)

    # Converte valores numéricos entre 0 e 1 para %
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # multiplica apenas se todos os valores estiverem no intervalo [0,1]
            if ((df[col] >= 0) & (df[col] <= 1)).all():
                df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
            else:
                df[col] = df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

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
        ("results/classificacao_holdout.csv", "photos/classificacao_holdout.png"),
        ("results/classificacao_crossval.csv", "photos/classificacao_crossval.png"),
        ("results/regressao_holdout.csv", "photos/regressao_holdout.png"),
        ("results/regressao_crossval.csv", "photos/regressao_crossval.png"),
    ]

    for csv_file, img_file in arquivos:
        csv_to_table_image(csv_file, img_file)
        print(f"✅ {img_file} gerada a partir de {csv_file}")
