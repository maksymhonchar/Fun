using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;

namespace OOP_Lab1_Gonchar_Maxim_KP51
{
    /// <summary>
    /// Class to help user build an appropriate log file.
    /// <para>printMsg(string contents) - write message into console.</para>
    /// <para>writeMsg(string contents) - write message only into the log.</para>
    /// <para>printAndWriteMsg(string contents) - write message into console and save it into the log.</para>
    /// </summary>
    class Logger
    {
        internal string loggerVersion = "v0.0.1";
        private string logFilePath;
        private string userName;
        private string promptSystem;
        private string promptPlayer;

        public Logger()
        {
            this.logFilePath = this.formLogFilePath("log.txt");
            this.userName = System.Security.Principal.WindowsIdentity.GetCurrent().Name;
            this.promptSystem = userName + "@system:~ $ ";
            this.promptPlayer = userName + "@player:~ $ ";
            this.printGreetings();
        }

        /// <summary>
        /// Form an appropriate path for the log file, that can be used for writing a log.
        /// </summary>
        /// <param name="logFileName">Actual log filename.</param>
        /// <returns>Full log path.</returns>
        private string formLogFilePath(string logFileName)
        {
            // Get a folder to save a log file.
            string[] dirsNames = Directory.GetCurrentDirectory().Split('\\');
            IEnumerable<string> dirsList = dirsNames.Take(dirsNames.Length - 2);
            string logFilePath = "";
            foreach (var item in dirsList)
            {
                logFilePath += item + @"\";
            }
            
            // Check if file with name "d.m.yyyy.log" already exists.
            // If it does - save it into *_v{N}.log, where {N} is a number. 
            int dummyCounter = 0;
            string possibleName = string.Format(@"{0:d/M/yyyy}.log", DateTime.Now);
            while (File.Exists(logFilePath + possibleName))
            {
                possibleName = String.Format(@"\{0:d/M/yyyy}_v{1}.log", DateTime.Now, ++dummyCounter);
            }
            
            // Return a file path with appropriate filename.
            return logFilePath + possibleName;
        }

        /// <summary>
        /// Print a greetings to the user. Display info about logger creation, user, log filepath.
        /// </summary>
        private void printGreetings()
        {
            Console.WriteLine("------------------------------------------------");
            Console.WriteLine("{0}Logger{1} has been created.", this.promptSystem, this.loggerVersion);
            Console.WriteLine("{0}You are using a program under {1} account.", this.promptSystem, this.userName);
            Console.WriteLine("{0}A log file will be saved to following path: {1}.", this.promptSystem, this.logFilePath);
            Console.WriteLine("------------------------------------------------");
        }

        /// <summary>
        /// Print a message to the console and write it into the .log file as an admin.
        /// </summary>
        /// <param name="contents">Message to be printed and written to the log.</param>
        public void printMsgSystem(string contents)
        {
            Console.WriteLine(promptSystem + contents);
        }

        /// <summary>
        /// Print a message to the console and write it into the .log file as an user.
        /// </summary>
        /// <param name="contents">Message to be printed and written to the log.</param>
        public void printMsgPlayer(string contents)
        {
            Console.WriteLine(promptPlayer + contents);
        }

        /// <summary>
        /// Write a message into the .log file.
        /// </summary>
        /// <param name="contents">Message to be written.</param>
        public void writeMsg(string contents)
        {
            File.WriteAllText(this.logFilePath, contents);
        }
    }
}
