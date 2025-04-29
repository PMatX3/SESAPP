css = '''
<style>
.chat-message-container {
            max-width: 800px;
            margin: 0 auto;
        }
        
'''

bot_template = '''
<div class="chat-message bot fade-in p-4 bg-gray-100 rounded-lg shadow-sm mb-4">
                <div class="flex space-x-4">
                    <div class="flex-shrink-0">
                        <div class="h-10 w-10 rounded-full bg-indigo-600 flex items-center justify-center text-white">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                            </svg>
                        </div>
                    </div>
                    <div class="flex-grow">
                        <div class="font-medium text-gray-900 mb-1">Assistant</div>
                        <div class="text-gray-700">{{MSG}}</div>
                    </div>
                </div>
            </div>
'''

user_template = '''
<div class="chat-message user fade-in p-4 bg-indigo-50 rounded-lg shadow-sm mb-4">
                <div class="flex space-x-4">
                    <div class="flex-shrink-0">
                        <div class="h-10 w-10 rounded-full bg-gray-700 flex items-center justify-center text-white">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                        </div>
                    </div>
                    <div class="flex-grow">
                        <div class="font-medium text-gray-900 mb-1">You</div>
                        <div class="text-gray-700">{{MSG}}</div>
                    </div>
                </div>
            </div>
'''