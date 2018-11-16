<h3>Ееее лайки </h3>
Ставит лайки на всех каментах на стене <br />

<h4>Что нужно изменить в коде</h4>
1) Установить свой айди - line 57, поле [user_id] в [main()] <br />
2) Установить свой токен юзера - line 18, аргумент [access_token] в [auth()] <br />
3) Установить адрес страницы, куда лить лайки - line 69, аргумент в [vkUser.addLikes_wall] в [main()] <br />

<h4>Чтобы узнать свой authkey нужно:</h4>
1) перейти по <a href='https://oauth.vk.com/authorize?client_id=5521254&scope=notify,friends,photos,pages,status,offers,wall,messages,offline&redirect_uri=blank.html&display=popup&response_type=token'>ссылке</a> и дать доступ приложению [maxdve testapp] <br />
2) скопировать адресную строку. <br />
Ее формат такой: <br />
https://oauth.vk.com/blank.html#access_token=AUTHENTIFICATION_KEY&expires_in=0&user_id=YOUR_USER_ID <br />
Пример моей адресной строки: <br />
<code>https://oauth.vk.com/blank.html#access_token=33c09331a3eeb53331e2e14asdasd132163699ff7560dd3ed5cf8f1677fd752052345sadfa57b723&expires_in=0&user_id=291823738</code> <br />
3) access_token - это то, что вам нужно!
