# src 1 (old, 2017): https://habr.com/ru/post/326986/

- Разработка Facebook
- В двух словах, GraphQL это синтаксис, который описывает как запрашивать данные, и, в основном, используется клиентом для загрузки данных с сервера.
- GraphQL имеет три основные характеристики:
    - Позволяет клиенту точно указать, какие данные ему нужны.
    - Облегчает агрегацию данных из нескольких источников.
    - Использует систему типов для описания данных.
- Facebook придумал концептуально простое решение: вместо того, чтобы иметь множество "глупых" endpoint, лучше иметь один "умный" endpoint, который будет способен работать со сложными запросами и придавать данным такую форму, какую запрашивает клиент.
- Фактически, слой GraphQL находится между клиентом и одним или несколькими источниками данных; он принимает запросы клиентов и возвращает необходимые данные в соответствии с переданными инструкциями. 
- Необходимо всего два компонента чтобы начать:
    1. Сервер GraphQL для обработки запросов к API
    2. Клиент GraphQL, который будет подключаться к endpoint.
        - Apollo-client - позволяет выполнять запросы GraphQL в браузере
        - GraphiQL - браузерная IDE для запросов к ендпоинтам GraphQL
- Компоненты Graph QL:
1. схема (schema)
2. запросы (queries)
    query getConcretePost($id : String) {
        posts(id: $id) { # это массив, отображается конкретный пост
            title
            body
            author { # мы может пойти глубже
                name
                avatarUrl
                profileUrl
            }
        }
    }
3. распознаватели (resolvers) - объясняет что делать серверу GraphQL с входящим запросом

# src 2: https://towardsdatascience.com/graphql-grafana-and-dash-7aa96d940f1b

- The main difference between REST and GraphQL is that RESTful APIs have multiple endpoints that return fixed data structures whereas a GraphQL server only exposes a single endpoint and returns flexible data structures.
- Query example:
`
query {
  books {
    id
    title
    author
    isbn
    price  } 
}
`
