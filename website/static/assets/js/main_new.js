/*
	Eventually by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

(function() {

	"use strict";

	var	$body = document.querySelector('body');

	// Methods/polyfills.

		// classList | (c) @remy | github.com/remy/polyfills | rem.mit-license.org
			!function(){function t(t){this.el=t;for(var n=t.className.replace(/^\s+|\s+$/g,"").split(/\s+/),i=0;i<n.length;i++)e.call(this,n[i])}function n(t,n,i){Object.defineProperty?Object.defineProperty(t,n,{get:i}):t.__defineGetter__(n,i)}if(!("undefined"==typeof window.Element||"classList"in document.documentElement)){var i=Array.prototype,e=i.push,s=i.splice,o=i.join;t.prototype={add:function(t){this.contains(t)||(e.call(this,t),this.el.className=this.toString())},contains:function(t){return-1!=this.el.className.indexOf(t)},item:function(t){return this[t]||null},remove:function(t){if(this.contains(t)){for(var n=0;n<this.length&&this[n]!=t;n++);s.call(this,n,1),this.el.className=this.toString()}},toString:function(){return o.call(this," ")},toggle:function(t){return this.contains(t)?this.remove(t):this.add(t),this.contains(t)}},window.DOMTokenList=t,n(Element.prototype,"classList",function(){return new t(this)})}}();

		// canUse
			window.canUse=function(p){if(!window._canUse)window._canUse=document.createElement("div");var e=window._canUse.style,up=p.charAt(0).toUpperCase()+p.slice(1);return p in e||"Moz"+up in e||"Webkit"+up in e||"O"+up in e||"ms"+up in e};

		// window.addEventListener
			(function(){if("addEventListener"in window)return;window.addEventListener=function(type,f){window.attachEvent("on"+type,f)}})();

	// Play initial animations on page load.
		window.addEventListener('load', function() {
			window.setTimeout(function() {
				$body.classList.remove('is-preload');
			}, 100);
		});

	// Slideshow Background.
		(function() {

			// Settings.
				var settings = {

					// Images (in the format of 'url': 'alignment').
						images: {
							'images/bg01.jpg': 'center',
							'images/bg02.jpg': 'center',
							'images/bg03.jpg': 'center'
						},

					// Delay.
						delay: 6000

				};

			// Vars.
				var	pos = 0, lastPos = 0,
					$wrapper, $bgs = [], $bg,
					k, v;

			// Create BG wrapper, BGs.
				$wrapper = document.createElement('div');
					$wrapper.id = 'bg';
					$body.appendChild($wrapper);

				for (k in settings.images) {

					// Create BG.
						$bg = document.createElement('div');
							$bg.style.backgroundImage = 'url("' + k + '")';
							$bg.style.backgroundPosition = settings.images[k];
							$wrapper.appendChild($bg);

					// Add it to array.
						$bgs.push($bg);

				}

			// Main loop.
				$bgs[pos].classList.add('visible');
				$bgs[pos].classList.add('top');

				// Bail if we only have a single BG or the client doesn't support transitions.
					if ($bgs.length == 1
					||	!canUse('transition'))
						return;

				window.setInterval(function() {

					lastPos = pos;
					pos++;

					// Wrap to beginning if necessary.
						if (pos >= $bgs.length)
							pos = 0;

					// Swap top images.
						$bgs[lastPos].classList.remove('top');
						$bgs[pos].classList.add('visible');
						$bgs[pos].classList.add('top');

					// Hide last image after a short delay.
						window.setTimeout(function() {
							$bgs[lastPos].classList.remove('visible');
						}, settings.delay / 2);

				}, settings.delay);

		})();
		(function() {



			// Settings.

				var settings = {



					// Images (in the format of 'url': 'alignment').

						images: {

							'images/bg01.jpg': 'center',

							'images/bg02.jpg': 'center',

							'images/bg03.jpg': 'center'

						},



					// Delay.

						delay: 6000



				};



			// Vars.

				var	pos = 0, lastPos = 0,

					$wrapper, $bgs = [], $bg,

					k, v;



			// Create BG wrapper, BGs.

				$wrapper = document.createElement('div');

					$wrapper.id = 'bg';

					$body.appendChild($wrapper);



				for (k in settings.images) {



					// Create BG.

						$bg = document.createElement('div');

							$bg.style.backgroundImage = 'url("' + k + '")';

							$bg.style.backgroundPosition = settings.images[k];

							$wrapper.appendChild($bg);



					// Add it to array.

						$bgs.push($bg);



				}



			// Main loop.

				$bgs[pos].classList.add('visible');

				$bgs[pos].classList.add('top');



				// Bail if we only have a single BG or the client doesn't support transitions.

					if ($bgs.length == 1

					||	!canUse('transition'))

						return;



				window.setInterval(function() {



					lastPos = pos;

					pos++;



					// Wrap to beginning if necessary.

						if (pos >= $bgs.length)

							pos = 0;



					// Swap top images.

						$bgs[lastPos].classList.remove('top');

						$bgs[pos].classList.add('visible');

						$bgs[pos].classList.add('top');



					// Hide last image after a short delay.

						window.setTimeout(function() {

							$bgs[lastPos].classList.remove('visible');

						}, settings.delay / 2);



				}, settings.delay);



		})();


	// Signup Form.
		(function() {

			// Vars.
				var $form = document.querySelectorAll('#signup-form')[0],
					$submit = document.querySelectorAll('#signup-form input[type="submit"]')[0],
					$message;

			// Bail if addEventListener isn't supported.
				if (!('addEventListener' in $form))
					return;

			// Message.
				$message = document.createElement('span');
					$message.classList.add('message');
					$form.appendChild($message);

				$message._show = function(type, text) {

					$message.innerHTML = text;
					$message.classList.add(type);
					$message.classList.add('visible');

					window.setTimeout(function() {
						$message._hide();
					}, 3000);

				};

				$message._hide = function() {
					$message.classList.remove('visible');
				};

			// Events.
			// Note: If you're *not* using AJAX, get rid of this event listener.
				$form.addEventListener('submit', function(event) {

					event.stopPropagation();
					event.preventDefault();

					// Hide message.
						$message._hide();

					// Disable submit.
						$submit.disabled = true;

					// Process form.
					// Note: Doesn't actually do anything yet (other than report back with a "thank you"),
					// but there's enough here to piece together a working AJAX submission call that does.
						window.setTimeout(function() {

							// Reset form.
								$form.reset();

							// Enable submit.
								$submit.disabled = false;

							// Show message.
								$message._show('success', 'Thank you!');
								//$message._show('failure', 'Something went wrong. Please try again.');

						}, 750);

				});

		})();

})();
var background = {}
  
background.initializr = function (){
  
  var $this = this;
   

 
  //option
  $this.id = "background_css3";
  $this.style = {bubbles_color:"#fff",stroke_width:0, stroke_color :"black"};
  $this.bubbles_number = 30;
  $this.speed = [1500,8000]; //milliseconds
  $this.max_bubbles_height = $this.height;
  $this.shape = false // 1 : circle | 2 : triangle | 3 : rect | false :random
  
  if($("#"+$this.id).lenght > 0){
	$("#"+$this.id).remove();
  }
  $this.object = $("<div style='z-inde:-1;margin:0;padding:0; overflow:hidden;position:absolute;bottom:0' id='"+$this.id+"'> </div>'").appendTo("header");
  
  $this.ww = $(window).width()
  $this.wh = $(window).height()
  $this.width = $this.object.width($this.ww);
  $this.height = $this.object.height($this.wh);
  
  
  $("body").prepend("<style>.shape_background {transform-origin:center; width:80px; height:80px; background: "+$this.style.bubbles_color+"; position: absolute}</style>");
  
  
  for (i = 0; i < $this.bubbles_number; i++) {
	  $this.generate_bubbles()
  }
  
}





 background.generate_bubbles = function() {
   var $this = this;
   var base = $("<div class='shape_background'></div>");
   var shape_type = $this.shape ? $this.shape : Math.floor($this.rn(1,3));
   if(shape_type == 1) {
	 var bolla = base.css({borderRadius: "50%"})
   }else if (shape_type == 2){
	 var bolla = base.css({width:0, height:0, "border-style":"solid","border-width":"0 40px 69.3px 40px","border-color":"transparent transparent "+$this.style.bubbles_color+" transparent", background:"transparent"}); 
   }else{
	 var bolla = base; 
   }    
   var rn_size = $this.rn(.8,1.2);
   bolla.css({"transform":"scale("+rn_size+") rotate("+$this.rn(-360,360)+"deg)", top:$this.wh+100, left:$this.rn(-60, $this.ww+60)});        
   bolla.appendTo($this.object);
   bolla.transit({top: $this.rn($this.wh/2,$this.wh/2-60), "transform":"scale("+rn_size+") rotate("+$this.rn(-360,360)+"deg)", opacity: 0},$this.rn($this.speed[0],$this.speed[1]), function(){
	 $(this).remove();
	 $this.generate_bubbles();
   })
	 
  }


background.rn = function(from, to, arr) {
if(arr){
		return Math.random() * (to - from + 1) + from;
}else{
  return Math.floor(Math.random() * (to - from + 1) + from);
}
  }
background.initializr()