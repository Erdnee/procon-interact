# File-уудын жагсаалт

* admin-token.txt -> API-руу admin-ий хувиар request явуулахад хэрэглэх token
* init_api.sh -> талбайн мэдээлэл болон token-ий мэдээллийг ачааллах script
* init-field.json -> талбайн мэдээллийг агуулах JSON file
* init-token.json -> token-ий мэдээллийг агуулах JSON file
* procon.jar -> API-г ажлуулах server-ийн jar file

# Ашиглалт

## procon server нь доорхи request-үүдэд response явуулна

### admin-control

* startgame -> тоглоомыг эхлүүлнэ 
* pausegame -> тоглоомыг түр зогсооно
* stopgame  -> тоглоомыг зогсооно
* initgame -> тоглоомын мэдээллийг хадгална
* inittoken -> багуудын token-ий мэдээллийг хадгална
### player-control

* board -> board-ийн GUI-ийг харах
* move -> agent-р нүүдэл хийлгэх
* status -> тоглоомын мэдээллийг авах

## Тоглоомыг тоглоход бэлэн болгох
1. ```sh
    $ java -jar procon.jar
    ```
    командыг ажлуулна.
2. ```sh
    $ ./init_api.sh
    ```
    командыг ажлуулсанаар init-field.json, init-token.json file-ууд доторхи мэдээллүүдийг server хүлээн авна.