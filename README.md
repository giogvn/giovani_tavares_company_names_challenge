Este repositório contém a solução para o case de Data Science envolvendo recuperação de informação e embeddings.

***
# Conteúdo do Repositório

- `full_solution.ipynb`: contém toda a documentação e códigos utilizados para a escolha do melhor retriever a realizar a tarefa em questão. 
- `test_best_retriever.ipynb`: é o notebook que deve ser utilizado no conjunto de testes para a geração das métricas de performance desejadas (top 1 e top 5 para `razaosocial`, `nome_fantasia` e `cnpj`). Para isso, basta abrir o notebook e modificar o nome das variáveis da seção 1 para corresponder aos seus dados de teste, como modificar o *path* para apontar para seu arquivo de dados. Após isso, é importante rodar **todas** as suas células para gerar os resultados final.
- Os resultados de performance para os dados fornecidos no case estão na seção 5 do notebook `test_best_retriever.ipynb`. Para reproduzi-los, basta apontar a variável `test_file_path` na seção 1 do mesmo notebook para o caminho arquivo contendo os dados fornecidos e rodar todas as células, obtendo os resultados na seção 4.