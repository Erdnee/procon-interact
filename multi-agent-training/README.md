# file-уудын жагсаалт

* **grid_env.py** -> Тоглоомын environment-ийг tensor болгож хадгалаад компьютерт тооцоолол хийхэд нь хялбар болгож өгөх зорилготой. Өөрөөр хэлвэл хүнд гоё харагддаг environment-ийг simplify хийж шинэ энгийн environment үүсгэнэ.
* pickle өргөтгөлтэй file-ууд нь 2 агент зэрэг тоглуулж сургасан policy-г хадгалсан  
* field-info.json -> талбайн мэдээллийг хадгалсан json file 
* **training_multi_agents.py** -> 2 агент зэрэг тоглуулж сургах програм.

# Ашиглалт
1. **training_multi_agents.py** нь execute хийх боломжтой бөгөөд ажлуулахаас өмнө API-г ачааллах server-ийг ажлуулах ёстой.
2. Түүний дараа талбайн мэдээлэл, token-ий мэдээлэл 2-ийг илгээнэ. Server ажлуулах заавар нь interaction folder-ийн README.md file-д байгаа болно.
3. 
    ```sh
    $ python training_multi_agents.py
    ```
    командыг ажлуулж сургана.
