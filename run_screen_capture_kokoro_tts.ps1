# Screen Capture Kokoro TTS - Send Win+T Key Combination
# Sends Windows key + T to cycle through taskbar items

# Minimize the PowerShell window
try {
    Add-Type -TypeDefinition @"
        using System;
        using System.Runtime.InteropServices;
        public class Window {
            [DllImport("kernel32.dll")]
            public static extern IntPtr GetConsoleWindow();
            
            [DllImport("user32.dll")]
            public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
            
            public const int SW_MINIMIZE = 6;
            public const int SW_RESTORE = 9;
        }
"@
    
    $consoleWindow = [Window]::GetConsoleWindow()
    if ($consoleWindow -ne [IntPtr]::Zero) {
        [Window]::ShowWindow($consoleWindow, [Window]::SW_MINIMIZE)
    }
} catch {
    # If minimization fails, continue normally
}

Write-Host "🎯 Kokoro TTS Screen Capture Trigger" -ForegroundColor Cyan
Write-Host "📍 Location: $PSScriptRoot" -ForegroundColor Gray
Write-Host ""

try {
    # Send Windows key + T combination using Windows API
    Write-Host "⌨️  Sending Windows key + T combination..." -ForegroundColor Yellow
    
    # Define Windows API for key simulation
    Add-Type -TypeDefinition @"
        using System;
        using System.Runtime.InteropServices;
        public class Win32 {
            [DllImport("user32.dll")]
            public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
            
            public const int VK_LWIN = 0x5B;
            public const int VK_T = 0x54;
            public const uint KEYEVENTF_KEYUP = 0x0002;
        }
"@
    
    # Press Windows key down
    [Win32]::keybd_event([Win32]::VK_LWIN, 0, 0, [UIntPtr]::Zero)
    
    # Press T key down
    [Win32]::keybd_event([Win32]::VK_T, 0, 0, [UIntPtr]::Zero)
    
    # Release T key
    [Win32]::keybd_event([Win32]::VK_T, 0, [Win32]::KEYEVENTF_KEYUP, [UIntPtr]::Zero)
    
    # Release Windows key
    [Win32]::keybd_event([Win32]::VK_LWIN, 0, [Win32]::KEYEVENTF_KEYUP, [UIntPtr]::Zero)
    
    Write-Host ""
    Write-Host "✨ Key combination sent successfully!" -ForegroundColor Green
    Write-Host "🔄 Windows key + T should cycle through taskbar items" -ForegroundColor Gray
    
} catch {
    Write-Host ""
    Write-Host "❌ Error sending key combination: $($_.Exception.Message)" -ForegroundColor Red
    
    # Try alternative method using SendKeys with correct syntax
    try {
        Write-Host "🔄 Trying SendKeys alternative..." -ForegroundColor Yellow
        Add-Type -AssemblyName System.Windows.Forms
        
        # Simulate pressing Win key by focusing and using Ctrl+Esc (Start menu equivalent) + T
        [System.Windows.Forms.SendKeys]::SendWait("^{ESC}")
        Start-Sleep -Milliseconds 100
        [System.Windows.Forms.SendKeys]::SendWait("t")
        
        Write-Host "✨ Alternative SendKeys method succeeded!" -ForegroundColor Green
    } catch {
        Write-Host "❌ All methods failed: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Monitor clipboard for changes
Write-Host ""
Write-Host "👀 Monitoring clipboard for changes..." -ForegroundColor Yellow
Write-Host "📝 INSTRUCTION: Copy some text to clipboard to trigger TTS" -ForegroundColor Cyan
Write-Host "⏰ Waiting up to 60 seconds for new clipboard content..." -ForegroundColor Gray

try {
    # Get initial clipboard content with simple string comparison
    $initialClipboard = ""
    try {
        $initialClipboard = Get-Clipboard -ErrorAction SilentlyContinue
        if ($null -eq $initialClipboard) { 
            $initialClipboard = "" 
        } else {
            # Convert to string and normalize
            $initialClipboard = [string]$initialClipboard
        }
    } catch {
        $initialClipboard = ""
    }
    
    # Store initial content length for comparison
    $initialLength = $initialClipboard.Length
    
    $displayInitial = if ($initialClipboard.Length -gt 50) { 
        $initialClipboard.Substring(0, 50) + "..." 
    } else { 
        $initialClipboard 
    }
    Write-Host "📋 Initial clipboard: '$displayInitial' (Length: $initialLength)" -ForegroundColor Gray
    
    # Monitor for clipboard changes (timeout after 60 seconds)
    $timeout = 60
    $elapsed = 0
    $checkInterval = 1
    $dotCounter = 0
    $lastStatusTime = 0
    $changeDetected = $false
    
    do {
        Start-Sleep -Seconds $checkInterval
        $elapsed += $checkInterval
        $dotCounter += $checkInterval
        
        try {
            $currentClipboard = Get-Clipboard -ErrorAction SilentlyContinue
            if ($null -eq $currentClipboard) { 
                $currentClipboard = "" 
            } else {
                # Convert to string and normalize
                $currentClipboard = [string]$currentClipboard
            }
        } catch {
            $currentClipboard = ""
        }
        
        # Check for changes using multiple criteria
        $contentChanged = $false
        if ($currentClipboard -ne $initialClipboard) {
            $contentChanged = $true
        } elseif ($currentClipboard.Length -ne $initialLength) {
            $contentChanged = $true
        }
        
        if ($contentChanged) {
            $changeDetected = $true
            break
        }
        
        # Show progress dots every 3 seconds and status every 10 seconds
        if ($dotCounter -ge 3) {
            Write-Host "." -NoNewline -ForegroundColor Gray
            $dotCounter = 0
        }
        
        # Show status reminder every 10 seconds
        if ($elapsed - $lastStatusTime -ge 10) {
            Write-Host ""
            Write-Host "💡 Waiting for clipboard content... ($elapsed s / $timeout s)" -ForegroundColor Yellow
            Write-Host "🔍 Current: '$($currentClipboard.Substring(0, [Math]::Min(30, $currentClipboard.Length)))$(if($currentClipboard.Length -gt 30){'...'})'" -ForegroundColor DarkGray
            $lastStatusTime = $elapsed
        }
        
    } while ($elapsed -lt $timeout)
    
    Write-Host ""
    
    if ($changeDetected) {
        Write-Host "✅ Clipboard content changed!" -ForegroundColor Green
        
        $displayCurrent = if ($currentClipboard.Length -gt 100) { 
            $currentClipboard.Substring(0, 100) + "..." 
        } else { 
            $currentClipboard 
        }
        Write-Host "📋 New content: '$displayCurrent'" -ForegroundColor Cyan
        Write-Host "📏 Length: $($currentClipboard.Length) characters" -ForegroundColor Gray
        
        # Only run TTS if there's actually meaningful content
        if ($currentClipboard.Length -gt 0 -and $currentClipboard.Trim() -ne "") {
            # Run Kokoro TTS
            Write-Host ""
            Write-Host "🚀 Running Kokoro TTS..." -ForegroundColor Yellow
            
            # Get the script directory
            $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
            $batFilePath = Join-Path $scriptDir "run_kokoro_tts.bat"
            
            Write-Host "📁 Script directory: $scriptDir" -ForegroundColor Gray
            Write-Host "🔍 Looking for: $batFilePath" -ForegroundColor Gray
            
            if (Test-Path $batFilePath) {
                # Save current location
                $originalLocation = Get-Location
                
                try {
                    # Change to script directory
                    Set-Location $scriptDir
                    Write-Host "📂 Changed to directory: $(Get-Location)" -ForegroundColor Gray
                    
                    # Run the batch file
                    cmd /c "run_kokoro_tts.bat"
                    Write-Host "✅ Kokoro TTS execution completed!" -ForegroundColor Green
                } catch {
                    Write-Host "❌ Error running TTS: $($_.Exception.Message)" -ForegroundColor Red
                } finally {
                    # Always restore original location
                    Set-Location $originalLocation
                    Write-Host "🔄 Restored to original directory: $(Get-Location)" -ForegroundColor Gray
                }
            } else {
                Write-Host "❌ run_kokoro_tts.bat not found at: $batFilePath" -ForegroundColor Red
                Write-Host "💡 Make sure the batch file exists in the same directory as this script" -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠️  Empty or whitespace-only clipboard content - skipping TTS" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⏰ Timeout: No clipboard changes detected within $timeout seconds" -ForegroundColor Yellow
        Write-Host "💡 To use: Copy some text while the script is running next time" -ForegroundColor Gray
    }
    
} catch {
    Write-Host "❌ Error monitoring clipboard: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🏁 Script completed." -ForegroundColor Cyan 