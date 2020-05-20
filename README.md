# Folder-уудын жагсаалт(Чухал Folder-уудыг **Bold**-р ялгав)

* **interaction** -> procon-ий API-г local орчинд ажлуулж тоглоомын өгөгдлийг ачааллах програмуудын folder
* **multi-agent-training** -> 2 agent-ийг зэрэг ажлуулж сургах програмуудын folder 
* **single-agent-training** -> agent-уудыг нэг нэгээр ажлуулж сургах програмуудын  folder
* **plot** -> agent-уудын өөр өөр аргууд дээр сурч байгаа үр дүнг харуулах графикийг үүсгэх програмуудын folder

* useful-pickle-files -> agent-уудын хийх үйлдлүүдийг заах policy-г хадгалах folder
* previous-tests -> single-agent-training-тэй ижилхэн нэг agent ажлуулж сургах програмуудын folder.  Хамгийн үр дүнтэй ажиллагаатай нэг програмыг сонгож single-agent-training folder-т хийсэн болно.

# Code-ийг ажлуулахаас өмнө анхаарах зүйлс

Програмууд дээр ашиглаж байгаа Python-ий module-уудыг *requirements.txt* дотор байгаа.

Тиймээс code-уудыг ажлуулахаас өмнө
    ```sh
    $ pip install -r requirements.txt
    ```
командыг дуудаж ажлуулах хэрэгтэй.