"""Hand-written base phrasings for the SYNTHETIC corpus.

Eight intents a consumer app's support inbox would plausibly see, in five
languages, six phrasings each. Index 0-3 are "known" wordings that appear on
day 1; index 4-5 are held back and only appear on day 2, so the assignment
step has to place wordings it has never embedded before. `change_address` is a
ninth intent that never appears on day 1 at all: it is the novel category the
margin test is supposed to refuse to force into an existing label.

Nothing here comes from any real dataset.
"""

LANGS = ["en", "es", "de", "fr", "pt"]

KNOWN = {
    "double_charge": {
        "en": ["charged twice", "double charge", "I was charged twice for one order", "my card was billed two times",
               "duplicate payment on my card", "you took the money twice"],
        "es": ["cobro doble", "me cobraron dos veces", "me han cobrado dos veces el mismo pedido", "cargo duplicado en mi tarjeta",
               "pago duplicado", "me descontaron el dinero dos veces"],
        "de": ["doppelt abgebucht", "doppelte Abbuchung", "mir wurde zweimal Geld abgebucht", "die Zahlung wurde zweimal von meiner Karte abgezogen",
               "Betrag zweimal belastet", "doppelte Zahlung für eine Bestellung"],
        "fr": ["débité deux fois", "double prélèvement", "j'ai été débité deux fois pour la même commande", "paiement en double sur ma carte",
               "on m'a facturé deux fois", "prélèvement effectué deux fois"],
        "pt": ["cobrança duplicada", "cobrado duas vezes", "fui cobrado duas vezes pelo mesmo pedido", "pagamento duplicado no cartão",
               "descontaram o valor duas vezes", "cobraram em dobro"],
    },
    "refund_missing": {
        "en": ["no refund", "refund not received", "still waiting for my refund", "you promised a refund two weeks ago and nothing came",
               "where is my money back", "refund never arrived"],
        "es": ["no me llegó el reembolso", "reembolso pendiente", "sigo esperando mi reembolso", "me prometieron un reembolso y no ha llegado nada",
               "dónde está mi devolución", "no he recibido la devolución del dinero"],
        "de": ["keine Rückerstattung", "Rückerstattung nicht erhalten", "ich warte immer noch auf mein Geld zurück", "die versprochene Erstattung ist nie angekommen",
               "wo bleibt meine Erstattung", "Geld nicht zurückerhalten"],
        "fr": ["pas de remboursement", "remboursement non reçu", "j'attends toujours mon remboursement", "on m'a promis un remboursement il y a deux semaines et rien",
               "où est mon remboursement", "je n'ai jamais été remboursé"],
        "pt": ["reembolso não recebido", "sem reembolso", "ainda estou esperando meu reembolso", "prometeram o estorno e até agora nada",
               "cadê meu dinheiro de volta", "o estorno nunca chegou"],
    },
    "login_failure": {
        "en": ["can't log in", "login not working", "it says my password is wrong but it's correct", "stuck on the login screen",
               "unable to sign in to my account", "verification code never arrives so I cannot log in"],
        "es": ["no puedo iniciar sesión", "no me deja entrar", "dice que mi contraseña es incorrecta pero es correcta", "me quedo atascado en la pantalla de inicio",
               "error al iniciar sesión", "no me llega el código para entrar"],
        "de": ["kann mich nicht anmelden", "Login funktioniert nicht", "es sagt mein Passwort ist falsch, aber es stimmt", "hänge beim Anmeldebildschirm fest",
               "Anmeldung schlägt fehl", "Bestätigungscode kommt nicht an, deshalb kein Login"],
        "fr": ["impossible de me connecter", "connexion impossible", "il dit que mon mot de passe est faux alors qu'il est bon", "bloqué sur l'écran de connexion",
               "je n'arrive pas à me connecter à mon compte", "le code de vérification n'arrive jamais"],
        "pt": ["não consigo entrar", "login não funciona", "diz que minha senha está errada mas está certa", "travado na tela de login",
               "erro ao fazer login", "o código de verificação não chega"],
    },
    "app_crash": {
        "en": ["app crashes", "keeps crashing", "the app closes as soon as I open it", "crashes every time I try to pay",
               "app freezes and shuts down", "it crashed again after the update"],
        "es": ["la app se cierra", "se bloquea todo el rato", "la aplicación se cierra nada más abrirla", "se cierra cada vez que intento pagar",
               "la app se congela y se cierra", "después de la actualización se cierra sola"],
        "de": ["App stürzt ab", "stürzt ständig ab", "die App schließt sich sofort nach dem Öffnen", "stürzt jedes Mal beim Bezahlen ab",
               "App friert ein und beendet sich", "seit dem Update stürzt sie ab"],
        "fr": ["l'appli plante", "plante tout le temps", "l'application se ferme dès que je l'ouvre", "plante à chaque fois que je veux payer",
               "l'appli se fige puis se ferme", "ça plante depuis la mise à jour"],
        "pt": ["o app trava", "fica fechando sozinho", "o aplicativo fecha assim que eu abro", "trava toda vez que tento pagar",
               "o app congela e fecha", "depois da atualização ele fecha sozinho"],
    },
    "late_delivery": {
        "en": ["order late", "package not arrived", "my order was supposed to arrive last week", "tracking hasn't moved in days",
               "still no delivery", "where is my parcel"],
        "es": ["pedido retrasado", "no ha llegado el paquete", "mi pedido tenía que llegar la semana pasada", "el seguimiento no se mueve desde hace días",
               "todavía sin entrega", "dónde está mi paquete"],
        "de": ["Bestellung verspätet", "Paket nicht angekommen", "meine Bestellung sollte letzte Woche kommen", "Sendungsverfolgung bewegt sich seit Tagen nicht",
               "immer noch keine Lieferung", "wo ist mein Paket"],
        "fr": ["commande en retard", "colis pas arrivé", "ma commande devait arriver la semaine dernière", "le suivi ne bouge plus depuis des jours",
               "toujours pas livré", "où est mon colis"],
        "pt": ["pedido atrasado", "o pacote não chegou", "meu pedido era para chegar semana passada", "o rastreio não atualiza há dias",
               "ainda não foi entregue", "cadê minha encomenda"],
    },
    "account_takeover": {
        "en": ["account hacked", "someone logged into my account", "I got an alert for a login I did not make", "my email was changed without me",
               "unknown device on my account", "someone is using my account"],
        "es": ["me hackearon la cuenta", "alguien entró en mi cuenta", "recibí un aviso de un inicio de sesión que no hice", "cambiaron mi correo sin mi permiso",
               "dispositivo desconocido en mi cuenta", "alguien está usando mi cuenta"],
        "de": ["Konto gehackt", "jemand hat sich in mein Konto eingeloggt", "ich habe eine Warnung zu einer Anmeldung bekommen, die nicht von mir war", "meine E-Mail wurde ohne mich geändert",
               "unbekanntes Gerät in meinem Konto", "jemand benutzt mein Konto"],
        "fr": ["compte piraté", "quelqu'un s'est connecté à mon compte", "j'ai reçu une alerte pour une connexion que je n'ai pas faite", "mon e-mail a été changé sans moi",
               "appareil inconnu sur mon compte", "quelqu'un utilise mon compte"],
        "pt": ["conta hackeada", "alguém entrou na minha conta", "recebi um alerta de um login que não fiz", "mudaram meu e-mail sem minha permissão",
               "dispositivo desconhecido na minha conta", "alguém está usando minha conta"],
    },
    "spam_messages": {
        "en": ["too many emails", "stop sending me promotions", "I get five marketing messages a day", "unsubscribe does not work",
               "spam notifications", "please stop the newsletters"],
        "es": ["demasiados correos", "dejen de enviarme promociones", "recibo cinco mensajes de publicidad al día", "darse de baja no funciona",
               "notificaciones de spam", "por favor paren los boletines"],
        "de": ["zu viele E-Mails", "hört auf mir Werbung zu schicken", "ich bekomme fünf Werbenachrichten am Tag", "Abmelden funktioniert nicht",
               "Spam-Benachrichtigungen", "bitte keine Newsletter mehr"],
        "fr": ["trop d'e-mails", "arrêtez de m'envoyer des promos", "je reçois cinq messages publicitaires par jour", "le désabonnement ne marche pas",
               "notifications de spam", "arrêtez les newsletters svp"],
        "pt": ["muitos e-mails", "parem de me mandar promoções", "recebo cinco mensagens de propaganda por dia", "cancelar inscrição não funciona",
               "notificações de spam", "por favor parem as newsletters"],
    },
    "cancel_subscription": {
        "en": ["cancel subscription", "how do I cancel", "I want to end my membership before it renews", "cancel my plan please",
               "stop my monthly subscription", "where is the cancel button"],
        "es": ["cancelar suscripción", "cómo cancelo", "quiero terminar mi membresía antes de que se renueve", "cancelen mi plan por favor",
               "quiero dar de baja mi suscripción mensual", "dónde está el botón para cancelar"],
        "de": ["Abo kündigen", "wie kündige ich", "ich möchte meine Mitgliedschaft vor der Verlängerung beenden", "bitte meinen Tarif kündigen",
               "monatliches Abo beenden", "wo ist der Kündigungsbutton"],
        "fr": ["résilier l'abonnement", "comment résilier", "je veux arrêter mon abonnement avant le renouvellement", "annulez mon forfait svp",
               "stopper mon abonnement mensuel", "où est le bouton pour résilier"],
        "pt": ["cancelar assinatura", "como cancelo", "quero encerrar minha assinatura antes de renovar", "cancelem meu plano por favor",
               "parar minha assinatura mensal", "onde fica o botão de cancelar"],
    },
}

NOVEL = {
    "change_address": {
        "en": ["change delivery address", "wrong address on my order", "I moved and need to update the shipping address",
               "can you send it to a different address", "update my address", "the address on the order is old"],
        "es": ["cambiar dirección de entrega", "dirección equivocada en mi pedido", "me mudé y necesito actualizar la dirección de envío",
               "pueden enviarlo a otra dirección", "actualizar mi dirección", "la dirección del pedido es antigua"],
        "de": ["Lieferadresse ändern", "falsche Adresse in meiner Bestellung", "ich bin umgezogen und muss die Lieferadresse ändern",
               "könnt ihr es an eine andere Adresse schicken", "meine Adresse aktualisieren", "die Adresse in der Bestellung ist alt"],
        "fr": ["changer l'adresse de livraison", "mauvaise adresse sur ma commande", "j'ai déménagé et je dois changer l'adresse de livraison",
               "pouvez-vous l'envoyer à une autre adresse", "mettre à jour mon adresse", "l'adresse de la commande est ancienne"],
        "pt": ["mudar endereço de entrega", "endereço errado no meu pedido", "me mudei e preciso atualizar o endereço de entrega",
               "podem enviar para outro endereço", "atualizar meu endereço", "o endereço do pedido está antigo"],
    },
}

# Content-free rows every inbox gets. Labelled `noise`.
NOISE = ["ok", "test", "asdf", "?", "hello", "thanks", "gracias", "danke", "merci", "obrigado", "hi", "...", "lol", "xd"]

KNOWN_IDX = [0, 1, 2, 3]   # seen on day 1
HELD_IDX = [4, 5]          # only on day 2
