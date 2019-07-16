var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function() {
  $messages.mCustomScrollbar();
  setTimeout(function() {
    //fakeMessage();
  }, 100);
});





function insertMessage() {
  msg1 = $('.form-first-name').val();
  msg2 = $('.form-last-number').val();
  msg3 = $('.form-last-zhusu').val();
  msg4 = $('.form-about-xianbingshi').val();
  msg5 = $('.form-about-jiwangshi').val();
  msg = msg1 + ','+ msg2 +','+ msg3 +','+ msg4 +','+ msg5
  console.info(msg)

  if ($.trim(msg) == '') {
    return false;
  }
  $('.message-input').val(null);
	interact(msg);

}

$('.btn btn-next').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
})


function interact(message){
	$.post('/message', {
		msg: message,
	}).done(function(reply) {
    $('.message.loading').remove();
    $('<div class="message new"><figure class="avatar"><img src="/static/res/easybot.png" /></figure>' + reply['text'] + '</div>').appendTo($('.mCSB_container')).addClass('new');
		}).fail(function() {
				alert('error calling function');
				});
}
